#include "sparkpipe/spark_nvme_tier.h"

#include <string.h>

// The mechanics behind the contract in spark_nvme_tier.h. Three tables, all
// carved from one caller-provided blob at init (runtime/arena.h pattern: one
// allocation, no malloc afterwards):
//
//   SLOTS     one per block the budget buys; slot index IS the device record,
//             so the offset is multiplication and there is no second allocator
//             to disagree with the first about what is free.
//   STAGING   pinned DMA buffers, double-buffered at minimum: one fills from
//             the drive while the other's contents are consumed upstairs, so
//             the drive never waits on consumption and consumption never waits
//             on the drive.
//   PENDING   the lookahead queue, kept sorted by earliest deadline so Pump's
//             issue loop is "take from the front while staging is free".
//
// EVICTION IS A CLOCK, and that is a load-bearing choice. A full LRU scan
// over a budget of millions of records costs microseconds-to-milliseconds on
// the path that admits a write-back, and an exact ordering buys nothing when
// the evicted bytes are re-fetchable: the cost of a wrong victim is one
// re-read, not a stall. Second-chance gives recency approximated by a single
// bit, O(1) amortised, and the generation counter makes eviction safe against
// DMA already in flight - a read landing in a recycled slot is waste to be
// discarded, never data to be believed.

#define NVME_TIER_SLOT_EMPTY 0u
#define NVME_TIER_SLOT_PRESENT 1u    /* on the drive, nothing moving */
#define NVME_TIER_SLOT_FILLING 2u    /* a read into staging is in flight */
#define NVME_TIER_SLOT_READY 3u      /* landed in staging, awaiting Consume */

#define NVME_TIER_STAGING_FREE 0u
#define NVME_TIER_STAGING_FILLING 1u
#define NVME_TIER_STAGING_READY 2u

#define NVME_TIER_HOLDER_NONE 0u
#define NVME_TIER_HOLDER_PREFETCH 1u
#define NVME_TIER_HOLDER_DEMAND 2u

typedef struct NvmeTierSlot
{
	uint64_t content_hash;         /* 0 until first publish */
	uint64_t last_use;             /* monotonic tick, for observability */
	uint32_t next_in_bucket;
	uint32_t next_free;            /* free-list link while EMPTY */
	uint32_t generation;           /* bumped on every recycle */
	uint32_t need_by_step;         /* earliest deadline anyone has stated */
	uint32_t issued_step;          /* when the in-flight read was submitted */
	uint32_t pin_count;
	uint32_t staging_index;        /* while FILLING or READY */
	uint8_t state;
	uint8_t referenced;            /* the clock's second-chance bit */
	uint8_t queued;                /* sits in the pending queue */
	uint8_t reserved0;
}
NvmeTierSlot;

typedef struct NvmeTierStagingState
{
	uint64_t ticket;
	uint32_t slot;
	uint32_t generation;           /* of the slot at issue time */
	uint32_t need_by_step;
	uint32_t issued_step;
	uint8_t state;
	uint8_t holder;                /* PREFETCH is preemptible, DEMAND is not */
	uint8_t reserved0;
	uint8_t reserved1;
}
NvmeTierStagingState;

typedef struct NvmeTierPendingEntry
{
	uint32_t slot;
	uint32_t generation;           /* of the slot at enqueue time */
	uint32_t need_by_step;
	uint32_t order;                /* FIFO tie-break inside one deadline */
}
NvmeTierPendingEntry;

typedef struct NvmeTierPendingQueue
{
	NvmeTierPendingEntry *entries;
	uint32_t count;
	uint32_t capacity;
	uint32_t next_order;
	uint32_t reserved0;
}
NvmeTierPendingQueue;

static uint32_t NvmeTierSlotCountForBudget(
	const SparkNvmeTierConfiguration *configuration)
{
	uint64_t count;
	if ( configuration->block_bytes == 0u )
		return(0u);
	count = configuration->budget_bytes / configuration->block_bytes;
	if ( count > 0xfffffffeULL )
		count = 0xfffffffeULL;
	return((uint32_t)count);
}

static uint64_t NvmeTierAlignU64(uint64_t value, uint64_t alignment)
{
	return((value + alignment - 1u) & ~(alignment - 1u));
}

uint64_t SparkNvmeTierTableBytes(
	const SparkNvmeTierConfiguration *configuration)
{
	uint64_t total;
	uint32_t slot_count;
	if ( configuration == 0 )
		return(0u);
	slot_count = NvmeTierSlotCountForBudget(configuration);
	if ( slot_count == 0u )
		return(0u);
	total = 0u;
	total = NvmeTierAlignU64(total + (uint64_t)slot_count * sizeof(NvmeTierSlot),8u);
	total = NvmeTierAlignU64(total + (uint64_t)configuration->hash_bucket_count * sizeof(uint32_t),8u);
	total = NvmeTierAlignU64(total + sizeof(NvmeTierPendingQueue)
		+ (uint64_t)configuration->pending_capacity * sizeof(NvmeTierPendingEntry),8u);
	total = NvmeTierAlignU64(total + (uint64_t)configuration->staging_buffer_count * sizeof(NvmeTierStagingState),8u);
	return(total);
}

static uint32_t NvmeTierLookup(
	const SparkNvmeTier *tier,
	uint64_t content_hash)
{
	uint32_t walk;
	const NvmeTierSlot *slots = (const NvmeTierSlot *)tier->slots;
	if ( content_hash == 0u )
		return(SPARK_NVME_TIER_NO_SLOT);
	walk = tier->buckets[content_hash % tier->configuration.hash_bucket_count];
	while ( walk != SPARK_NVME_TIER_NO_SLOT )
	{
		if ( slots[walk].state != NVME_TIER_SLOT_EMPTY
			&& slots[walk].content_hash == content_hash )
			return(walk);
		walk = slots[walk].next_in_bucket;
	}
	return(SPARK_NVME_TIER_NO_SLOT);
}

static void NvmeTierBucketInsert(SparkNvmeTier *tier, uint32_t slot_index)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t bucket = (uint32_t)(slots[slot_index].content_hash
		% tier->configuration.hash_bucket_count);
	slots[slot_index].next_in_bucket = tier->buckets[bucket];
	tier->buckets[bucket] = slot_index;
}

static void NvmeTierBucketRemove(SparkNvmeTier *tier, uint32_t slot_index)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t bucket = (uint32_t)(slots[slot_index].content_hash
		% tier->configuration.hash_bucket_count);
	uint32_t walk = tier->buckets[bucket];
	uint32_t previous = SPARK_NVME_TIER_NO_SLOT;
	while ( walk != SPARK_NVME_TIER_NO_SLOT && walk != slot_index )
	{
		previous = walk;
		walk = slots[walk].next_in_bucket;
	}
	if ( walk == slot_index )
	{
		if ( previous == SPARK_NVME_TIER_NO_SLOT )
			tier->buckets[bucket] = slots[slot_index].next_in_bucket;
		else
			slots[previous].next_in_bucket = slots[slot_index].next_in_bucket;
	}
	slots[slot_index].next_in_bucket = SPARK_NVME_TIER_NO_SLOT;
}

SparkStatus SparkNvmeTierInitialize(
	SparkNvmeTier *tier,
	const SparkNvmeTierConfiguration *configuration,
	const SparkNvmeTierDevice *device,
	void *tables,
	void *staging)
{
	uint32_t slot_count,index;
	uint8_t *cursor;
	NvmeTierSlot *slots;
	NvmeTierStagingState *staging_states;
	NvmeTierPendingQueue *queue;
	uint64_t bytes_per_step;
	if ( tier == 0 || configuration == 0 || device == 0
		|| tables == 0 || staging == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_NVME_TIER_ABI_VERSION
		|| configuration->descriptor_bytes != SPARK_NVME_TIER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	slot_count = NvmeTierSlotCountForBudget(configuration);
	if ( slot_count == 0u
		|| configuration->hash_bucket_count == 0u
		|| configuration->pending_capacity == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	// A block payload that is not a whole number of O_DIRECT granules forces
	// the driver into buffered I/O; rejecting the configuration beats finding
	// the copy at bring-up.
	if ( ( configuration->block_bytes % SPARK_NVME_TIER_IO_ALIGNMENT_BYTES ) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->staging_buffer_count < 2u
		|| configuration->staging_buffer_count > SPARK_NVME_TIER_MAX_STAGING_BUFFERS
		|| configuration->demand_reserve_buffers >= configuration->staging_buffer_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->device_bytes_per_second == 0u
		|| configuration->step_time_microseconds == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( device->submit_read == 0 || device->poll_read == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	// The driver DMAs into this memory; a misaligned staging blob is an
	// O_DIRECT failure at runtime, so it is a failure here instead.
	if ( ( (uintptr_t)staging % SPARK_NVME_TIER_IO_ALIGNMENT_BYTES ) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(tier,0,sizeof(*tier));
	tier->configuration = *configuration;
	tier->device = *device;
	tier->slot_count = slot_count;
	tier->staging = (uint8_t *)staging;
	cursor = (uint8_t *)tables;
	tier->slots = cursor;
	cursor += NvmeTierAlignU64((uint64_t)slot_count * sizeof(NvmeTierSlot),8u);
	tier->buckets = (uint32_t *)(void *)cursor;
	cursor += NvmeTierAlignU64((uint64_t)configuration->hash_bucket_count * sizeof(uint32_t),8u);
	tier->pending = cursor;
	cursor += NvmeTierAlignU64((uint64_t)configuration->pending_capacity * sizeof(NvmeTierPendingEntry),8u);
	tier->staging_state = cursor;
	slots = (NvmeTierSlot *)tier->slots;
	for ( index = 0u; index < slot_count; ++index )
	{
		slots[index].content_hash = 0u;
		slots[index].state = NVME_TIER_SLOT_EMPTY;
		slots[index].generation = 0u;
		slots[index].need_by_step = 0xffffffffu;
		slots[index].next_in_bucket = SPARK_NVME_TIER_NO_SLOT;
		slots[index].next_free = index + 1u;
		slots[index].staging_index = SPARK_NVME_TIER_NO_SLOT;
	}
	slots[slot_count - 1u].next_free = SPARK_NVME_TIER_NO_SLOT;
	tier->free_head = 0u;
	for ( index = 0u; index < configuration->hash_bucket_count; ++index )
		tier->buckets[index] = SPARK_NVME_TIER_NO_SLOT;
	staging_states = (NvmeTierStagingState *)tier->staging_state;
	for ( index = 0u; index < configuration->staging_buffer_count; ++index )
	{
		staging_states[index].state = NVME_TIER_STAGING_FREE;
		staging_states[index].holder = NVME_TIER_HOLDER_NONE;
		staging_states[index].slot = SPARK_NVME_TIER_NO_SLOT;
	}
	queue = (NvmeTierPendingQueue *)tier->pending;
	// The queue header lives at the head of its own region; the entries follow.
	queue->entries = (NvmeTierPendingEntry *)(void *)( queue + 1 );
	queue->count = 0u;
	queue->capacity = configuration->pending_capacity;
	queue->next_order = 0u;
	tier->tick = 1u;
	tier->clock_hand = 0u;
	// The two conversions the planner needs every call, computed once:
	// bandwidth and step time to bytes-per-step, and block size to the number
	// of steps a read takes. A division per need per step is the kind of
	// arithmetic that never shows in a profile until it does.
	bytes_per_step = ( configuration->device_bytes_per_second
		* configuration->step_time_microseconds ) / 1000000ULL;
	if ( bytes_per_step == 0u )
		bytes_per_step = 1u;
	if ( bytes_per_step > 0xffffffffULL )
		bytes_per_step = 0xffffffffULL;
	tier->bytes_per_step = (uint32_t)bytes_per_step;
	tier->transfer_steps = (uint32_t)(
		( configuration->block_bytes + bytes_per_step - 1u ) / bytes_per_step );
	if ( tier->transfer_steps == 0u )
		tier->transfer_steps = 1u;
	tier->statistics.slot_count = slot_count;
	return(SPARK_STATUS_OK);
}

// Release a staging buffer back to FREE, detaching it from its slot. The slot
// drops to PRESENT when it still names this buffer: the on-drive record
// outlives every staging cycle, which is what makes dropping a prefetch cheap.
static void NvmeTierStagingRelease(SparkNvmeTier *tier, uint32_t staging_index)
{
	NvmeTierStagingState *staging_states = (NvmeTierStagingState *)tier->staging_state;
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t slot_index = staging_states[staging_index].slot;
	if ( slot_index != SPARK_NVME_TIER_NO_SLOT
		&& slots[slot_index].staging_index == staging_index
		&& ( slots[slot_index].state == NVME_TIER_SLOT_FILLING
			|| slots[slot_index].state == NVME_TIER_SLOT_READY ) )
	{
		slots[slot_index].state = NVME_TIER_SLOT_PRESENT;
		slots[slot_index].staging_index = SPARK_NVME_TIER_NO_SLOT;
	}
	staging_states[staging_index].state = NVME_TIER_STAGING_FREE;
	staging_states[staging_index].holder = NVME_TIER_HOLDER_NONE;
	staging_states[staging_index].slot = SPARK_NVME_TIER_NO_SLOT;
}

// The eviction clock. Two revolutions at worst: the first clears second-chance
// bits, the second is guaranteed to meet a victim it cleared - unless every
// slot is one it must not touch:
//
//   pinned        - admission promised this block to a scheduled sequence.
//   demand-held   - its staging buffer is the decode path's data; evicting it
//                   under the reader is the one eviction that WOULD stall a
//                   step, which is the thing this tier exists to prevent.
//   filling, no cancel - the device owns the staging buffer until the read
//                   lands; recycling the slot now would let late DMA land in a
//                   buffer already reused for someone else.
//
// A single revolution cannot evict anything once every slot has its grace
// bit set - found by the test's ninth publish into an 8-record tier. Two
// revolutions with no victim means the tier is genuinely wedged (every record
// pinned or demand-held) and the caller hears BUSY - loud, instead of the
// silent alternative of evicting a block a scheduled sequence is about to
// read.
static uint32_t NvmeTierClockEvict(SparkNvmeTier *tier)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	NvmeTierStagingState *staging_states = (NvmeTierStagingState *)tier->staging_state;
	uint32_t probe;
	for ( probe = 0u; probe < 2u * tier->slot_count; ++probe )
	{
		uint32_t index = tier->clock_hand;
		NvmeTierSlot *slot = &slots[index];
		tier->clock_hand = ( tier->clock_hand + 1u ) % tier->slot_count;
		if ( slot->state == NVME_TIER_SLOT_EMPTY )
			continue;
		if ( slot->pin_count != 0u )
		{
			tier->statistics.pinned_eviction_skips++;
			continue;
		}
		if ( slot->state == NVME_TIER_SLOT_FILLING || slot->state == NVME_TIER_SLOT_READY )
		{
			NvmeTierStagingState *held = &staging_states[slot->staging_index];
			if ( held->holder == NVME_TIER_HOLDER_DEMAND )
				continue;
			if ( slot->state == NVME_TIER_SLOT_FILLING && tier->device.cancel_read == 0 )
				continue;
		}
		if ( slot->referenced != 0u )
		{
			slot->referenced = 0u;
			continue;
		}
		// Victim. Cheap by construction: no write-back, no flush - KV is
		// recomputable and the source tier may still hold it, so the record is
		// simply forgotten and its generation bumps, which is also what makes
		// any read still in flight for it harmless.
		if ( slot->state == NVME_TIER_SLOT_FILLING )
		{
			NvmeTierStagingState *held = &staging_states[slot->staging_index];
			tier->device.cancel_read(tier->device.context,held->ticket);
			NvmeTierStagingRelease(tier,slot->staging_index);
		}
		else if ( slot->state == NVME_TIER_SLOT_READY )
		{
			if ( staging_states[slot->staging_index].holder == NVME_TIER_HOLDER_PREFETCH )
				tier->statistics.prefetch_dropped++;
			NvmeTierStagingRelease(tier,slot->staging_index);
		}
		slot->queued = 0u;   /* any pending entry for it dies by generation check */
		NvmeTierBucketRemove(tier,index);
		slot->content_hash = 0u;
		slot->state = NVME_TIER_SLOT_EMPTY;
		slot->generation++;
		slot->pin_count = 0u;
		slot->next_free = tier->free_head;
		tier->free_head = index;
		tier->slots_in_use--;
		tier->statistics.evictions++;
		return(index);
	}
	return(SPARK_NVME_TIER_NO_SLOT);
}

static uint32_t NvmeTierSlotAcquire(SparkNvmeTier *tier)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t index;
	if ( tier->free_head != SPARK_NVME_TIER_NO_SLOT )
	{
		index = tier->free_head;
		tier->free_head = slots[index].next_free;
		slots[index].next_free = SPARK_NVME_TIER_NO_SLOT;
		return(index);
	}
	if ( NvmeTierClockEvict(tier) == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_NVME_TIER_NO_SLOT);
	// The clock pushed its victim onto the free list.
	index = tier->free_head;
	tier->free_head = slots[index].next_free;
	slots[index].next_free = SPARK_NVME_TIER_NO_SLOT;
	return(index);
}

SparkStatus SparkNvmeTierPublish(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	uint64_t *device_offset_out)
{
	NvmeTierSlot *slots;
	uint32_t slot_index;
	if ( tier == 0 || content_hash == 0u || device_offset_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index != SPARK_NVME_TIER_NO_SLOT )
	{
		// Idempotent: the write-back path may re-publish after a retry, and a
		// second record for the same hash is how one block ends up at two
		// offsets with reads split between them.
		slots[slot_index].last_use = tier->tick++;
		slots[slot_index].referenced = 1u;
		*device_offset_out = tier->configuration.base_offset
			+ (uint64_t)slot_index * tier->configuration.block_bytes;
		return(SPARK_STATUS_OK);
	}
	slot_index = NvmeTierSlotAcquire(tier);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_STATUS_BUSY);
	slots[slot_index].content_hash = content_hash;
	slots[slot_index].state = NVME_TIER_SLOT_PRESENT;
	slots[slot_index].last_use = tier->tick++;
	slots[slot_index].referenced = 1u;   /* just written: one grace period */
	slots[slot_index].pin_count = 0u;
	slots[slot_index].queued = 0u;
	slots[slot_index].need_by_step = 0xffffffffu;
	slots[slot_index].staging_index = SPARK_NVME_TIER_NO_SLOT;
	NvmeTierBucketInsert(tier,slot_index);
	tier->slots_in_use++;
	tier->statistics.publishes++;
	tier->statistics.slots_in_use = tier->slots_in_use;
	*device_offset_out = tier->configuration.base_offset
		+ (uint64_t)slot_index * tier->configuration.block_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierOffsetOf(
	const SparkNvmeTier *tier,
	uint64_t content_hash,
	uint64_t *device_offset_out)
{
	uint32_t slot_index;
	if ( tier == 0 || device_offset_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_STATUS_NOT_FOUND);
	*device_offset_out = tier->configuration.base_offset
		+ (uint64_t)slot_index * tier->configuration.block_bytes;
	return(SPARK_STATUS_OK);
}

// -- the pending queue ------------------------------------------------------------
//
// Sorted by (need_by_step, insertion order), earliest first, so Pump issues
// from the head. Insertion keeps the order; capacity is small (the lookahead
// window bounds how much can usefully be queued), so a memmove per insert is
// the right cost and a heap would be ceremony.

static int32_t NvmeTierPendingFind(
	const NvmeTierPendingQueue *queue,
	uint32_t slot_index)
{
	uint32_t index;
	for ( index = 0u; index < queue->count; ++index )
		if ( queue->entries[index].slot == slot_index )
			return((int32_t)index);
	return(-1);
}

static void NvmeTierPendingInsert(
	NvmeTierPendingQueue *queue,
	uint32_t slot_index,
	uint32_t generation,
	uint32_t need_by_step)
{
	uint32_t position = queue->count;
	while ( position != 0u )
	{
		const NvmeTierPendingEntry *before = &queue->entries[position - 1u];
		if ( before->need_by_step < need_by_step
			|| ( before->need_by_step == need_by_step
				&& before->order < queue->next_order ) )
			break;
		position--;
	}
	if ( position != queue->count )
		memmove(&queue->entries[position + 1u],&queue->entries[position],
			( queue->count - position ) * sizeof(NvmeTierPendingEntry));
	queue->entries[position].slot = slot_index;
	queue->entries[position].generation = generation;
	queue->entries[position].need_by_step = need_by_step;
	queue->entries[position].order = queue->next_order++;
	queue->count++;
}

static void NvmeTierPendingRemoveAt(NvmeTierPendingQueue *queue, uint32_t position)
{
	if ( position + 1u < queue->count )
		memmove(&queue->entries[position],&queue->entries[position + 1u],
			( queue->count - position - 1u ) * sizeof(NvmeTierPendingEntry));
	queue->count--;
}

// An earlier deadline for a queued block pulls it forward in the queue; the
// schedule got tighter, and the I/O order should say so.
static void NvmeTierPendingTighten(
	NvmeTierPendingQueue *queue,
	uint32_t position,
	uint32_t need_by_step)
{
	uint32_t slot,generation,order;
	if ( queue->entries[position].need_by_step <= need_by_step )
		return;
	slot = queue->entries[position].slot;
	generation = queue->entries[position].generation;
	order = queue->entries[position].order;
	NvmeTierPendingRemoveAt(queue,position);
	{
		uint32_t insert_at = queue->count;
		while ( insert_at != 0u )
		{
			const NvmeTierPendingEntry *before = &queue->entries[insert_at - 1u];
			if ( before->need_by_step < need_by_step
				|| ( before->need_by_step == need_by_step && before->order < order ) )
				break;
			insert_at--;
		}
		if ( insert_at != queue->count )
			memmove(&queue->entries[insert_at + 1u],&queue->entries[insert_at],
				( queue->count - insert_at ) * sizeof(NvmeTierPendingEntry));
		queue->entries[insert_at].slot = slot;
		queue->entries[insert_at].generation = generation;
		queue->entries[insert_at].need_by_step = need_by_step;
		queue->entries[insert_at].order = order;
		queue->count++;
	}
}

SparkStatus SparkNvmeTierPlanLookahead(
	SparkNvmeTier *tier,
	const SparkNvmeTierNeed *needs,
	uint32_t need_count,
	uint32_t step_now,
	SparkNvmeTierPlanReport *report_out)
{
	NvmeTierSlot *slots;
	NvmeTierPendingQueue *queue;
	SparkNvmeTierPlanReport report;
	uint32_t index;
	if ( tier == 0 || ( needs == 0 && need_count != 0u ) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&report,0,sizeof(report));
	slots = (NvmeTierSlot *)tier->slots;
	queue = (NvmeTierPendingQueue *)tier->pending;
	for ( index = 0u; index < need_count; ++index )
	{
		uint64_t hash = needs[index].content_hash;
		uint32_t need_by = needs[index].need_by_step;
		uint32_t slot_index = NvmeTierLookup(tier,hash);
		NvmeTierSlot *slot;
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		{
			// Not on the drive. Admission must see this: planning around an
			// absent block as if it were queued is how a sequence gets admitted
			// warm and starts cold.
			report.absent_count++;
			continue;
		}
		slot = &slots[slot_index];
		slot->last_use = tier->tick++;
		slot->referenced = 1u;      /* someone will need it: the clock should know */
		if ( need_by < slot->need_by_step || slot->state == NVME_TIER_SLOT_PRESENT )
			slot->need_by_step = need_by;
		if ( slot->state == NVME_TIER_SLOT_READY )
		{
			report.already_ready_count++;
			continue;
		}
		if ( slot->state == NVME_TIER_SLOT_FILLING )
		{
			NvmeTierStagingState *held
				= &((NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
			if ( need_by < held->need_by_step )
				held->need_by_step = need_by;
			report.already_inflight_count++;
			continue;
		}
		// PRESENT. Queued already: an earlier deadline pulls it forward.
		{
			int32_t position = NvmeTierPendingFind(queue,slot_index);
			if ( position >= 0 )
			{
				NvmeTierPendingTighten(queue,(uint32_t)position,need_by);
				report.queued_count++;
				continue;
			}
		}
		// Late risk is reported, not hidden: a need closer than the transfer
		// time cannot arrive on schedule, and the caller prefers to know
		// before admission rather than at the stalled step. The read is still
		// queued - late bytes beat no bytes by exactly the recompute time.
		if ( need_by <= step_now
			|| need_by - step_now < tier->transfer_steps )
			report.late_risk_count++;
		if ( queue->count >= queue->capacity )
		{
			report.queue_full_count++;
			continue;
		}
		slot->queued = 1u;
		NvmeTierPendingInsert(queue,slot_index,slot->generation,need_by);
		report.queued_count++;
	}
	if ( report_out != 0 )
		*report_out = report;
	return(SPARK_STATUS_OK);
}

// Staging for a demand load. The ordering is the priority contract in code:
//
//   1. a FREE buffer;
//   2. a buffer holding a landed PREFETCH nobody has consumed - the bytes are
//      on the drive, so dropping them costs one future re-read at most;
//   3. a buffer a PREFETCH read is still filling, cancelled - the drive loses
//      some positioning, the decode step keeps its budget;
//   4. nothing: every buffer holds DEMAND data awaiting consumption, which is
//      a sizing bug and is counted as one (demand_stalls) rather than hidden
//      as latency.
//
// Pinned slots are skipped in 2 and 3: admission pinned them precisely because
// a scheduled sequence cannot afford to re-fetch them.
// A displaced prefetch is re-queued, not forgotten: the need that motivated it
// is still in the schedule, and dropping it outright would convert a cheap
// buffer hand-off into a demand load later - exactly the critical-path read
// the prefetch existed to prevent.
static void NvmeTierPrefetchRequeue(SparkNvmeTier *tier, uint32_t slot_index, uint32_t need_by_step)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	NvmeTierPendingQueue *queue = (NvmeTierPendingQueue *)tier->pending;
	if ( slots[slot_index].queued != 0u || queue->count >= queue->capacity )
		return;
	slots[slot_index].queued = 1u;
	NvmeTierPendingInsert(queue,slot_index,slots[slot_index].generation,need_by_step);
}

static uint32_t NvmeTierStagingAcquireForDemand(SparkNvmeTier *tier)
{
	NvmeTierStagingState *staging_states = (NvmeTierStagingState *)tier->staging_state;
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t count = tier->configuration.staging_buffer_count;
	uint32_t index;
	for ( index = 0u; index < count; ++index )
		if ( staging_states[index].state == NVME_TIER_STAGING_FREE )
			return(index);
	for ( index = 0u; index < count; ++index )
	{
		uint32_t slot_index,deadline;
		if ( staging_states[index].state != NVME_TIER_STAGING_READY
			|| staging_states[index].holder != NVME_TIER_HOLDER_PREFETCH )
			continue;
		slot_index = staging_states[index].slot;
		if ( slots[slot_index].pin_count != 0u )
			continue;
		deadline = staging_states[index].need_by_step;
		tier->statistics.prefetch_preemptions++;
		tier->statistics.prefetch_dropped++;
		NvmeTierStagingRelease(tier,index);
		NvmeTierPrefetchRequeue(tier,slot_index,deadline);
		return(index);
	}
	if ( tier->device.cancel_read != 0 )
	{
		uint32_t best = SPARK_NVME_TIER_NO_SLOT;
		uint32_t best_deadline = 0u;
		for ( index = 0u; index < count; ++index )
		{
			if ( staging_states[index].state != NVME_TIER_STAGING_FILLING
				|| staging_states[index].holder != NVME_TIER_HOLDER_PREFETCH )
				continue;
			if ( slots[staging_states[index].slot].pin_count != 0u )
				continue;
			// Cancel the prefetch needed FURTHEST in the future: it has the
			// most time to be re-queued and re-fetched before its deadline.
			if ( best == SPARK_NVME_TIER_NO_SLOT
				|| staging_states[index].need_by_step > best_deadline )
			{
				best = index;
				best_deadline = staging_states[index].need_by_step;
			}
		}
		if ( best != SPARK_NVME_TIER_NO_SLOT )
		{
			uint32_t slot_index = staging_states[best].slot;
			uint32_t deadline = staging_states[best].need_by_step;
			tier->device.cancel_read(tier->device.context,staging_states[best].ticket);
			tier->statistics.prefetch_preemptions++;
			NvmeTierStagingRelease(tier,best);
			NvmeTierPrefetchRequeue(tier,slot_index,deadline);
			return(best);
		}
	}
	tier->statistics.demand_stalls++;
	return(SPARK_NVME_TIER_NO_SLOT);
}

static SparkStatus NvmeTierIssueRead(
	SparkNvmeTier *tier,
	uint32_t slot_index,
	uint32_t staging_index,
	uint8_t holder,
	uint32_t need_by_step,
	uint32_t step_now)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	NvmeTierStagingState *staging_states = (NvmeTierStagingState *)tier->staging_state;
	uint64_t ticket = 0u;
	SparkStatus status;
	status = tier->device.submit_read(tier->device.context,
		tier->configuration.base_offset
			+ (uint64_t)slot_index * tier->configuration.block_bytes,
		tier->staging + (uint64_t)staging_index * tier->configuration.block_bytes,
		tier->configuration.block_bytes,&ticket);
	if ( status != SPARK_STATUS_OK )
	{
		// The device refused; leave the slot PRESENT so a later pump retries,
		// and the buffer free so nothing else pays for the refusal.
		tier->statistics.io_errors++;
		NvmeTierStagingRelease(tier,staging_index);
		return(status);
	}
	staging_states[staging_index].ticket = ticket;
	staging_states[staging_index].slot = slot_index;
	staging_states[staging_index].generation = slots[slot_index].generation;
	staging_states[staging_index].need_by_step = need_by_step;
	staging_states[staging_index].issued_step = step_now;
	staging_states[staging_index].state = NVME_TIER_STAGING_FILLING;
	staging_states[staging_index].holder = holder;
	slots[slot_index].state = NVME_TIER_SLOT_FILLING;
	slots[slot_index].staging_index = staging_index;
	slots[slot_index].issued_step = step_now;
	tier->statistics.read_bytes += tier->configuration.block_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierRequestDemand(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	uint32_t step_now,
	SparkNvmeTierDemandResult *result_out)
{
	NvmeTierSlot *slots;
	uint32_t slot_index;
	NvmeTierSlot *slot;
	if ( tier == 0 || result_out == 0 || content_hash == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(result_out,0,sizeof(*result_out));
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
	{
		// The genuine miss, and the only one: the bytes are not on this tier
		// at all, so somebody recomputes. Counted, because a rising line here
		// is the write-back path falling behind, not bad luck.
		tier->statistics.demand_misses++;
		result_out->state = SPARK_NVME_TIER_DEMAND_MISS;
		return(SPARK_STATUS_OK);
	}
	slot = &slots[slot_index];
	slot->last_use = tier->tick++;
	slot->referenced = 1u;
	if ( slot->state == NVME_TIER_SLOT_READY )
	{
		NvmeTierStagingState *held
			= &((NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
		// Promote to DEMAND-held. Without this, a second demand load could
		// preempt-steal this buffer between the hit and the caller's Consume,
		// handing back a pointer whose bytes are about to be overwritten.
		held->holder = NVME_TIER_HOLDER_DEMAND;
		tier->statistics.demand_hits++;
		result_out->state = SPARK_NVME_TIER_DEMAND_READY;
		result_out->staging_pointer = tier->staging
			+ (uint64_t)slot->staging_index * tier->configuration.block_bytes;
		return(SPARK_STATUS_OK);
	}
	if ( slot->state == NVME_TIER_SLOT_FILLING )
	{
		// Join the read already moving - demand or prefetch, it does not
		// matter: the bytes arrive either way, and a second read of the same
		// record is the drive's worst case scheduled by hand.
		NvmeTierStagingState *held
			= &((NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
		held->holder = NVME_TIER_HOLDER_DEMAND;
		tier->statistics.demand_joins++;
		result_out->state = SPARK_NVME_TIER_DEMAND_IN_FLIGHT;
		return(SPARK_STATUS_OK);
	}
	// PRESENT. If the lookahead queued it, the queue loses it: demand is the
	// deadline now, and a queued prefetch that a demand load duplicates is
	// the inversion this whole design exists to prevent.
	if ( slot->queued != 0u )
	{
		NvmeTierPendingQueue *queue = (NvmeTierPendingQueue *)tier->pending;
		int32_t position = NvmeTierPendingFind(queue,slot_index);
		if ( position >= 0 )
			NvmeTierPendingRemoveAt(queue,(uint32_t)position);
		slot->queued = 0u;
	}
	{
		uint32_t staging_index = NvmeTierStagingAcquireForDemand(tier);
		SparkStatus status;
		if ( staging_index == SPARK_NVME_TIER_NO_SLOT )
			return(SPARK_STATUS_BUSY);
		status = NvmeTierIssueRead(tier,slot_index,staging_index,
			NVME_TIER_HOLDER_DEMAND,step_now,step_now);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	tier->statistics.demand_loads++;
	result_out->state = SPARK_NVME_TIER_DEMAND_STARTED;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierPump(SparkNvmeTier *tier, uint32_t step_now)
{
	NvmeTierSlot *slots;
	NvmeTierStagingState *staging_states;
	NvmeTierPendingQueue *queue;
	uint32_t index,free_count;
	if ( tier == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	staging_states = (NvmeTierStagingState *)tier->staging_state;
	queue = (NvmeTierPendingQueue *)tier->pending;
	// Completions first, issues second: a buffer freed by a landing can carry
	// the next read in the same pump, which is the double-buffer doing its job.
	for ( index = 0u; index < tier->configuration.staging_buffer_count; ++index )
	{
		NvmeTierStagingState *held = &staging_states[index];
		SparkStatus status;
		uint32_t slot_index;
		if ( held->state != NVME_TIER_STAGING_FILLING )
			continue;
		status = tier->device.poll_read(tier->device.context,held->ticket);
		if ( status == SPARK_STATUS_BUSY )
			continue;
		slot_index = held->slot;
		if ( status != SPARK_STATUS_OK )
		{
			tier->statistics.io_errors++;
			NvmeTierStagingRelease(tier,index);
			continue;
		}
		// The generation check is what makes cheap eviction safe: this read
		// was issued against a slot that may have been evicted and recycled
		// while the bytes were in flight. Landing them now would label one
		// block's bytes with another's hash.
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT
			|| slots[slot_index].state != NVME_TIER_SLOT_FILLING
			|| slots[slot_index].staging_index != index
			|| slots[slot_index].generation != held->generation )
		{
			tier->statistics.stale_completions++;
			NvmeTierStagingRelease(tier,index);
			continue;
		}
		held->state = NVME_TIER_STAGING_READY;
		slots[slot_index].state = NVME_TIER_SLOT_READY;
		if ( held->holder == NVME_TIER_HOLDER_PREFETCH )
		{
			tier->statistics.prefetch_landings++;
			if ( step_now > held->need_by_step )
				tier->statistics.prefetch_late_landings++;
		}
	}
	free_count = 0u;
	for ( index = 0u; index < tier->configuration.staging_buffer_count; ++index )
		if ( staging_states[index].state == NVME_TIER_STAGING_FREE )
			free_count++;
	// Prefetches issue only into staging ABOVE the demand reserve. The reserve
	// is the mechanism behind "prefetch never starves demand": a lookahead
	// queue deep enough to fill every buffer would turn the next miss into a
	// stall, so the last buffers are simply not prefetch's to take.
	while ( queue->count != 0u
		&& free_count > tier->configuration.demand_reserve_buffers )
	{
		NvmeTierPendingEntry head = queue->entries[0];
		uint32_t slot_index = head.slot;
		uint32_t staging_index = SPARK_NVME_TIER_NO_SLOT;
		NvmeTierPendingRemoveAt(queue,0u);
		if ( slot_index >= tier->slot_count
			|| slots[slot_index].generation != head.generation
			|| slots[slot_index].state != NVME_TIER_SLOT_PRESENT
			|| slots[slot_index].queued == 0u )
			continue;   /* evicted or already claimed while it waited */
		slots[slot_index].queued = 0u;
		for ( index = 0u; index < tier->configuration.staging_buffer_count; ++index )
			if ( staging_states[index].state == NVME_TIER_STAGING_FREE )
			{
				staging_index = index;
				break;
			}
		if ( staging_index == SPARK_NVME_TIER_NO_SLOT )
			break;      /* reserve miscounted: stop rather than take it */
		staging_states[staging_index].state = NVME_TIER_STAGING_FILLING;  /* claim */
		staging_states[staging_index].holder = NVME_TIER_HOLDER_PREFETCH;
		staging_states[staging_index].slot = SPARK_NVME_TIER_NO_SLOT;
		free_count--;
		if ( NvmeTierIssueRead(tier,slot_index,staging_index,
				NVME_TIER_HOLDER_PREFETCH,head.need_by_step,step_now) != SPARK_STATUS_OK )
			continue;
		tier->statistics.prefetch_issues++;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierConsume(SparkNvmeTier *tier, uint64_t content_hash)
{
	NvmeTierSlot *slots;
	uint32_t slot_index;
	if ( tier == 0 || content_hash == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_STATUS_NOT_FOUND);
	if ( slots[slot_index].state != NVME_TIER_SLOT_READY )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots[slot_index].last_use = tier->tick++;
	slots[slot_index].referenced = 1u;
	// The record stays PRESENT: consumption copies the bytes upstairs, it does
	// not move them. Eviction, and only eviction, reclaims the drive.
	NvmeTierStagingRelease(tier,slots[slot_index].staging_index);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierPin(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	int32_t pin)
{
	uint32_t slot_index;
	NvmeTierSlot *slots;
	if ( tier == 0 || content_hash == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_STATUS_NOT_FOUND);
	if ( pin )
		slots[slot_index].pin_count++;
	else if ( slots[slot_index].pin_count != 0u )
		slots[slot_index].pin_count--;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierWillBeResidentBy(
	const SparkNvmeTier *tier,
	const uint64_t *content_hashes,
	uint32_t hash_count,
	uint32_t step_now,
	uint32_t step_deadline,
	SparkNvmeTierResidencyAssessment *assessment_out)
{
	const NvmeTierSlot *slots;
	uint32_t index,confident;
	if ( tier == 0 || ( content_hashes == 0 && hash_count != 0u )
		|| assessment_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(assessment_out,0,sizeof(*assessment_out));
	slots = (const NvmeTierSlot *)tier->slots;
	for ( index = 0u; index < hash_count; ++index )
	{
		uint32_t slot_index = NvmeTierLookup(tier,content_hashes[index]);
		const NvmeTierSlot *slot;
		uint64_t eta_steps;
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		{
			assessment_out->absent_count++;
			continue;
		}
		slot = &slots[slot_index];
		if ( slot->state == NVME_TIER_SLOT_READY )
		{
			assessment_out->ready_count++;
			continue;
		}
		if ( slot->state == NVME_TIER_SLOT_FILLING )
		{
			// Worst-case ETA: the full transfer from now. Using the issued
			// step to discount elapsed time assumes the drive kept pace, and
			// an admission decision made on an optimistic ETA is how a warm
			// sequence starts cold.
			eta_steps = tier->transfer_steps;
			if ( (uint64_t)step_now + eta_steps <= step_deadline )
				assessment_out->inflight_confident_count++;
			else
				assessment_out->late_count++;
			continue;
		}
		// PRESENT: the read still has to be issued, so the queue in front of
		// it is part of the ETA. One staging buffer's worth of queue drains
		// per landing, so queue depth approximates the wait in steps.
		{
			const NvmeTierPendingQueue *queue = (const NvmeTierPendingQueue *)tier->pending;
			eta_steps = (uint64_t)tier->transfer_steps * ( 1u + queue->count );
		}
		if ( (uint64_t)step_now + eta_steps <= step_deadline )
			assessment_out->planned_confident_count++;
		else
			assessment_out->late_count++;
	}
	confident = assessment_out->ready_count
		+ assessment_out->inflight_confident_count
		+ assessment_out->planned_confident_count;
	if ( assessment_out->absent_count == 0u && assessment_out->late_count == 0u )
		assessment_out->confidence = SPARK_NVME_TIER_CONFIDENCE_ALL;
	else if ( confident == 0u )
		assessment_out->confidence = SPARK_NVME_TIER_CONFIDENCE_NONE;
	else
		assessment_out->confidence = SPARK_NVME_TIER_CONFIDENCE_PARTIAL;
	return(SPARK_STATUS_OK);
}

void SparkNvmeTierGetStatistics(
	const SparkNvmeTier *tier,
	SparkNvmeTierStatistics *statistics_out)
{
	if ( tier == 0 || statistics_out == 0 )
		return;
	*statistics_out = tier->statistics;
	statistics_out->slot_count = tier->slot_count;
	statistics_out->slots_in_use = tier->slots_in_use;
}

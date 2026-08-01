// Size-classed slab arena for comms paths that must not touch malloc after
// init.
//
// The ring work-dispatch and completion paths retained one heap copy per
// queued packet (calloc on enqueue, free on pop) and grew the resident decode
// payload with realloc on the dispatch path. On a pump that moves a packet
// per step per rank that is two allocator round trips per packet - unbounded
// latency in glibc's arena lock under contention, and fragmentation across a
// long-lived daemon. The arena takes one allocation at init, and every
// acquire/release after that is a free-list pop/push: O(1), no syscall, no
// lock, no fallback. Exhaustion is a loud NULL, never a quiet malloc.
//
// Storage layout is one backing block: every class owns a slot region plus a
// per-slot link array, both carved out of the same allocation, so destroy is
// a single free and no pointer leaves the block. The link array doubles as
// the in-use marker (state-pool pattern): releasing a slot that is not marked
// in use is a double free or a foreign pointer and is rejected, which is the
// ABA/stale-pointer safety a single-threaded pump needs - the free-list head
// cannot race, so a monotonic acquire generation per class is kept for
// observability rather than correctness.
//
// Threading contract: not internally synchronized. The one consumer today is
// the ring service backend work queue, which is pumped from the single
// service loop only (no pthread touches node/backend.c); a caller that adds a
// second pump thread must serialize acquire/release itself.
//
// Error codes are unique per site across the whole header so a return value
// identifies the call that produced it.

#ifndef SPARKPIPE_RUNTIME_ARENA_H
#define SPARKPIPE_RUNTIME_ARENA_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_ARENA_MAX_CLASS_COUNT 8u
// 16 bytes: a queued packet starts with u64 identity fields and is memcpy'd
// in bulk, so every slot stays friendly to 128-bit loads and aligned SIMD
// copies.
#define SPARK_ARENA_ALIGNMENT 16u
#define SPARK_ARENA_NO_SLOT 0xffffffffu
// A slot in use is marked distinctly from the free-list terminator, same as
// spark_state_pool.h: releasing into an EMPTY class writes the terminator
// into the slot's link, and if NO_SLOT also meant "in use" a double release
// would look legal.
#define SPARK_ARENA_IN_USE 0xfffffffeu

typedef struct SparkArenaClassDescriptor
{
	uint32_t slot_bytes;
	uint32_t slot_count;
} SparkArenaClassDescriptor;

typedef struct SparkArenaClass
{
	uint8_t *slots;
	uint32_t *links;
	uint32_t slot_bytes;
	uint32_t slot_count;
	uint32_t free_head;
	uint32_t free_count;
	uint32_t generation;
	uint32_t reserved0;
} SparkArenaClass;

typedef struct SparkArena
{
	uint8_t *backing;
	uint64_t backing_bytes;
	SparkArenaClass classes[SPARK_ARENA_MAX_CLASS_COUNT];
	uint32_t class_count;
	uint32_t reserved0;
} SparkArena;

static inline int32_t SparkArenaInitialize(
	SparkArena *arena,
	const SparkArenaClassDescriptor *descriptors,
	uint32_t class_count)
{
	uint64_t total_bytes;
	uint64_t region_offset;
	uint32_t class_index;
	uint32_t slot_index;
	if (arena == 0 || descriptors == 0 || class_count == 0u ||
		class_count > SPARK_ARENA_MAX_CLASS_COUNT)
		return -30901;
	memset(arena,0,sizeof(*arena));
	total_bytes = 0u;
	for (class_index = 0u; class_index < class_count; ++class_index)
	{
		uint64_t slot_bytes;
		uint64_t slot_count;
		slot_bytes = descriptors[class_index].slot_bytes;
		slot_count = descriptors[class_index].slot_count;
		if (slot_bytes == 0u || slot_count == 0u)
			return -30902;
		slot_bytes = (slot_bytes + (SPARK_ARENA_ALIGNMENT - 1u)) &
			~((uint64_t)SPARK_ARENA_ALIGNMENT - 1u);
		/* Acquire picks the FIRST class that fits, so the descriptor order
		   is the contract: ascending slot_bytes. A misordered table would
		   silently route small allocations into large slots. */
		if (class_index != 0u &&
			slot_bytes <=
				(uint64_t)descriptors[class_index - 1u].slot_bytes)
			return -30903;
		if (slot_count > (UINT64_MAX - total_bytes) /
				(slot_bytes + (uint64_t)sizeof(uint32_t)))
			return -30904;
		total_bytes += slot_count *
			(slot_bytes + (uint64_t)sizeof(uint32_t));
	}
	arena->backing = (uint8_t *)malloc((size_t)total_bytes);
	if (arena->backing == 0)
		return -30905;
	arena->backing_bytes = total_bytes;
	arena->class_count = class_count;
	/* Two passes: every slot region first (each a multiple of 16 bytes,
	   so every class stays 16-aligned relative to the backing), then the
	   per-slot link arrays (4-aligned by the same argument). Interleaving
	   a 4-byte link array between 16-byte slot regions would push every
	   later class off the alignment contract. */
	region_offset = 0u;
	for (class_index = 0u; class_index < class_count; ++class_index)
	{
		uint64_t slot_bytes;
		slot_bytes = (descriptors[class_index].slot_bytes +
				(SPARK_ARENA_ALIGNMENT - 1u)) &
			~((uint64_t)SPARK_ARENA_ALIGNMENT - 1u);
		arena->classes[class_index].slots = arena->backing + region_offset;
		region_offset += (uint64_t)descriptors[class_index].slot_count *
			slot_bytes;
	}
	for (class_index = 0u; class_index < class_count; ++class_index)
	{
		SparkArenaClass *arena_class;
		uint64_t slot_bytes;
		uint64_t slot_count;
		arena_class = &arena->classes[class_index];
		slot_bytes = (descriptors[class_index].slot_bytes +
				(SPARK_ARENA_ALIGNMENT - 1u)) &
			~((uint64_t)SPARK_ARENA_ALIGNMENT - 1u);
		slot_count = descriptors[class_index].slot_count;
		arena_class->links =
			(uint32_t *)(void *)(arena->backing + region_offset);
		region_offset += slot_count * (uint64_t)sizeof(uint32_t);
		arena_class->slot_bytes = (uint32_t)slot_bytes;
		arena_class->slot_count = (uint32_t)slot_count;
		arena_class->free_head = 0u;
		arena_class->free_count = (uint32_t)slot_count;
		arena_class->generation = 0u;
		for (slot_index = 0u; slot_index < slot_count; ++slot_index)
			arena_class->links[slot_index] = slot_index + 1u;
		arena_class->links[slot_count - 1u] = SPARK_ARENA_NO_SLOT;
	}
	return 0;
}

static inline void SparkArenaDestroy(
	SparkArena *arena)
{
	if (arena == 0)
		return;
	/* Zeroed-by-construction arenas (a state struct that never reached
	   arena init on a failed startup path) have backing == 0 and destroy
	   is a no-op for them. */
	free(arena->backing);
	memset(arena,0,sizeof(*arena));
}

static inline void *SparkArenaAcquire(
	SparkArena *arena,
	uint32_t bytes)
{
	uint32_t class_index;
	if (arena == 0 || bytes == 0u)
		return 0;
	for (class_index = 0u; class_index < arena->class_count; ++class_index)
	{
		SparkArenaClass *arena_class;
		uint32_t slot;
		arena_class = &arena->classes[class_index];
		if (bytes > arena_class->slot_bytes)
			continue;
		/* No fall-through to a larger class on exhaustion: silent promotion
		   is how a heap dies - the loud NULL tells the caller its sizing
		   was wrong instead of borrowing capacity it cannot count on. */
		if (arena_class->free_count == 0u)
			return 0;
		slot = arena_class->free_head;
		arena_class->free_head = arena_class->links[slot];
		arena_class->links[slot] = SPARK_ARENA_IN_USE;
		arena_class->free_count -= 1u;
		arena_class->generation += 1u;
		return arena_class->slots +
			((uint64_t)slot * arena_class->slot_bytes);
	}
	return 0;
}

static inline int32_t SparkArenaRelease(
	SparkArena *arena,
	void *pointer)
{
	uint32_t class_index;
	if (arena == 0 || pointer == 0)
		return -30906;
	for (class_index = 0u; class_index < arena->class_count; ++class_index)
	{
		SparkArenaClass *arena_class;
		uint64_t offset;
		uint32_t slot;
		arena_class = &arena->classes[class_index];
		if ((uint8_t *)pointer < arena_class->slots ||
			(uint8_t *)pointer >= arena_class->slots +
				((uint64_t)arena_class->slot_count *
				 arena_class->slot_bytes))
			continue;
		offset = (uint64_t)((uint8_t *)pointer - arena_class->slots);
		if (offset % arena_class->slot_bytes != 0u)
			return -30907;
		slot = (uint32_t)(offset / arena_class->slot_bytes);
		if (arena_class->links[slot] != SPARK_ARENA_IN_USE)
			return -30908;
		arena_class->links[slot] = arena_class->free_head;
		arena_class->free_head = slot;
		arena_class->free_count += 1u;
		return 0;
	}
	/* Inside no class: foreign pointer. */
	return -30909;
}

static inline uint64_t SparkArenaReservedBytes(
	const SparkArena *arena)
{
	if (arena == 0)
		return 0u;
	return arena->backing_bytes;
}

static inline uint32_t SparkArenaClassGeneration(
	const SparkArena *arena,
	uint32_t class_index)
{
	if (arena == 0 || class_index >= arena->class_count)
		return 0u;
	return arena->classes[class_index].generation;
}

#endif

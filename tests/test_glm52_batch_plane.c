#include "sparkpipe/spark_glm52_batch_sequence_table.h"
#include "sparkpipe/spark_glm52_expert_queue.h"
#include "sparkpipe/spark_glm52_jit_kv_pool.h"
#include "sparkpipe/spark_glm52_kv_dedup.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static SparkGlm52ExpertQueue test_queue;
static SparkGlm52JitKvPool test_pool;
static SparkGlm52BatchSequenceTable test_table;
static SparkGlm52KvDedup test_dedup;

static void SparkTestExpertQueueThresholdDeadlineAndOrder(void)
{
	SparkGlm52ExpertQueueConfiguration configuration;
	SparkGlm52ExpertQueueFiring firing;
	uint32_t row_index;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_EXPERT_QUEUE_ABI_VERSION;
	configuration.layer_count = 6u;
	configuration.expert_count = 256u;
	configuration.firing_threshold_rows = 4u;
	configuration.firing_deadline_ns = 1000000u;
	assert(SparkGlm52ExpertQueueInitialize(&test_queue,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,0u,&firing) == SPARK_STATUS_NOT_FOUND);
	for (row_index=0u; row_index<4u; row_index++)
		assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,2u,17u,1000u + row_index,100u + row_index) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,1u,200u,2000u,50u) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,2u,3u,3000u,60u) == SPARK_STATUS_OK);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,200u,&firing) == SPARK_STATUS_OK);
	assert(firing.layer_index == 2u && firing.expert_index == 17u && firing.row_count == 4u);
	assert(firing.row_ids[0u] == 1000u && firing.row_ids[3u] == 1003u);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,200u,&firing) == SPARK_STATUS_NOT_FOUND);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,50u + 1000000u,&firing) == SPARK_STATUS_OK);
	assert(firing.layer_index == 1u && firing.expert_index == 200u && firing.row_count == 1u);
	assert(firing.row_ids[0u] == 2000u);
	assert(SparkGlm52ExpertQueueNextFiring(&test_queue,60u + 1000000u,&firing) == SPARK_STATUS_OK);
	assert(firing.layer_index == 2u && firing.expert_index == 3u && firing.row_count == 1u);
	assert(test_queue.enqueued_row_count == 0u);
	assert(test_queue.firing_count == 3u && test_queue.fired_row_count == 6u);
	// Lazy free-list correctness: allocate past a fired-and-recycled batch and
	// confirm rows keep flowing without the eager init that used to touch 1M
	// entries. Overfill one slot beyond the emit cap and verify the cap holds
	// and the remainder stays queued with a correct advanced oldest arrival.
	{
		uint32_t bulk_index;
		SparkGlm52ExpertQueueFiring bulk;
		for (bulk_index=0u; bulk_index<SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS + 200u; ++bulk_index)
			assert(SparkGlm52ExpertQueueEnqueueRow(&test_queue,0u,0u,7000u + bulk_index,10u + bulk_index) == SPARK_STATUS_OK);
		assert(SparkGlm52ExpertQueueNextFiring(&test_queue,0u,&bulk) == SPARK_STATUS_OK);
		assert(bulk.row_count == SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS);
		assert(bulk.row_ids[0u] == 7000u);
		assert(test_queue.slots[0u][0u].count == 200u);
		assert(test_queue.slots[0u][0u].oldest_arrival_ns == 10u + SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS);
	}
}

static void SparkTestJitKvPoolPrefetchEvictionAndLateness(void)
{
	SparkGlm52JitKvPoolConfiguration configuration;
	uint32_t require_ids[2u];
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	configuration.fragment_capacity = 8u;
	configuration.dram_fragment_capacity = 2u;
	configuration.fragment_bytes = 1000000u;
	configuration.nvme_bytes_per_second = 1000000000u;
	assert(SparkGlm52JitKvPoolInitialize(&test_pool,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,0u,10u,0u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,1u,10u,1u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,2u,11u,0u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolAdmitFragment(&test_pool,3u,11u,1u,SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	require_ids[0u] = 0u;
	require_ids[1u] = 1u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,0u,require_ids,2u,5000000u) == SPARK_STATUS_OK);
	assert(test_pool.hit_count == 2u && test_pool.miss_count == 0u);
	require_ids[0u] = 2u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,0u,require_ids,1u,3000000u) == SPARK_STATUS_OK);
	assert(test_pool.miss_count == 1u && test_pool.stage_in_count == 1u && test_pool.stage_out_count == 1u);
	assert(test_pool.staging_in_count == 1u && test_pool.staging_out_count == 1u);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,2u) == 0u);
	assert(SparkGlm52JitKvPoolTick(&test_pool,3000000u) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,2u) == 1u);
	assert(test_pool.dram_resident_count == 2u && test_pool.late_count == 0u);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,0u) == 1u);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,1u) == 0u);
	require_ids[0u] = 3u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,3000000u,require_ids,1u,3500000u) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolTick(&test_pool,10000000u) == SPARK_STATUS_OK);
	assert(SparkGlm52JitKvPoolFragmentIsResident(&test_pool,3u) == 1u);
	assert(test_pool.late_count == 1u);
	require_ids[0u] = 0u;
	assert(SparkGlm52JitKvPoolRequireByEta(&test_pool,10000000u,require_ids,1u,10500000u) == SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkTestBatchSequenceTableLifecycleAndThreshold(void)
{
	SparkGlm52BatchSequenceTableConfiguration configuration;
	uint32_t first_handle,second_handle,first_index;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_BATCH_SEQUENCE_ABI_VERSION;
	configuration.sequence_capacity = 4u;
	configuration.lane_count = 8u;
	assert(SparkGlm52BatchSequenceTableInitialize(&test_table,&configuration) == SPARK_STATUS_OK);
	assert(SparkGlm52BatchSequenceTableAdmit(&test_table,900u,8192u,0u,128u,&first_handle) == SPARK_STATUS_OK);
	assert(SparkGlm52BatchSequenceTableAdmit(&test_table,901u,8192u,128u,128u,&second_handle) == SPARK_STATUS_OK);
	assert(test_table.active_count == 2u);
	assert(SparkGlm52BatchSequenceTableFiringThreshold(&test_table,8u,256u,1024u) == 1u);
	{
		uint32_t fill_index,scratch;
		for (fill_index=2u; fill_index<4u; fill_index++)
			assert(SparkGlm52BatchSequenceTableAdmit(&test_table,900u + fill_index,8192u,fill_index * 128u,128u,&scratch) == SPARK_STATUS_OK);
		assert(SparkGlm52BatchSequenceTableAdmit(&test_table,999u,8192u,512u,128u,&scratch) == SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	assert(SparkGlm52BatchSequenceTableFiringThreshold(&test_table,8u,256u,1024u) == 1u);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,first_handle) == SPARK_STATUS_OK);
	assert(test_table.active_count == 3u && test_table.awaiting_tool_count == 1u);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,first_handle) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52BatchSequenceTableBeginExchange(&test_table,first_handle,192u) == SPARK_STATUS_OK);
	first_index = (first_handle & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK);
	assert(test_table.sequences[first_index].exchange_number == 1u);
	assert(test_table.sequences[first_index].context_tokens == 8384u);
	assert(test_table.active_count == 4u && test_table.exchange_count == 5u);
	assert(SparkGlm52BatchSequenceTableComplete(&test_table,second_handle) == SPARK_STATUS_OK);
	assert(test_table.active_count == 3u && test_table.complete_count == 1u);
	// The stale handle now fails handle resolution, not just the state check:
	// the generation moved when the slot was freed, so a holdover handle can
	// never act on the slot's next occupant.
	assert(SparkGlm52BatchSequenceTableComplete(&test_table,second_handle) == SPARK_STATUS_NOT_FOUND);
	assert(SparkGlm52BatchSequenceTablePauseForTool(&test_table,second_handle) == SPARK_STATUS_NOT_FOUND);
	// The completed slot must be reclaimable: a capacity-4 table that has seen
	// completions keeps admitting under churn instead of leaking slots forever.
	{
		uint32_t churn_index,recycled,previous = second_handle;
		for (churn_index=0u; churn_index<64u; ++churn_index)
		{
			assert(SparkGlm52BatchSequenceTableAdmit(&test_table,5000u + churn_index,4089u,0u,64u,&recycled) == SPARK_STATUS_OK);
			assert((recycled & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK) ==
				(second_handle & SPARK_GLM52_BATCH_SEQUENCE_HANDLE_INDEX_MASK));
			assert(recycled != previous);
			previous = recycled;
			assert(SparkGlm52BatchSequenceTableComplete(&test_table,recycled) == SPARK_STATUS_OK);
		}
	}
}

static SparkGlm52JitKvPool stress_pool;

static void SparkTestJitKvPoolScaleAndBurst(void)
{
	SparkGlm52JitKvPoolConfiguration configuration;
	uint32_t fragment_index,require_id;
	uint64_t now_ns = 0u;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	configuration.fragment_capacity = 200000u;
	configuration.dram_fragment_capacity = 100000u;
	configuration.fragment_bytes = 442368u;
	configuration.nvme_bytes_per_second = 6000000000u;
	assert(SparkGlm52JitKvPoolInitialize(&stress_pool,&configuration) == SPARK_STATUS_OK);
	for (fragment_index=0u; fragment_index<200000u; fragment_index++)
		assert(SparkGlm52JitKvPoolAdmitFragment(&stress_pool,fragment_index,fragment_index / 32u,fragment_index % 32u,fragment_index < 100000u ? SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM : SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	assert(stress_pool.eviction_heap_count == 100000u);
	// Frozen-time burst: once the transfer ring is full of in-flight pairs the
	// NVMe is saturated and the require must REFUSE, never complete a transfer
	// whose done time is in the future, because that would mark a fragment
	// resident before its data physically arrived.
	{
		uint32_t refused = 0u;
		SparkStatus burst_status;
		for (fragment_index=0u; fragment_index<3000u; fragment_index++)
		{
			require_id = 100000u + fragment_index;
			burst_status = SparkGlm52JitKvPoolRequireByEta(&stress_pool,now_ns,&require_id,1u,now_ns + 20000000000u);
			if ( burst_status == SPARK_STATUS_CAPACITY_EXCEEDED )
			{
				refused += 1u;
				continue;
			}
			assert(burst_status == SPARK_STATUS_OK);
		}
		assert(refused != 0u);
		assert(stress_pool.overflow_drain_count == 0u);
		// Nothing staged at frozen time may report resident: no transfer's done
		// time has passed.
		assert(SparkGlm52JitKvPoolFragmentIsResident(&stress_pool,100000u) == 0u);
	}
	// Advancing-time burst: with real time passing each require, completed
	// transfers drain opportunistically and every require succeeds without any
	// early completion.
	{
		uint64_t transfer_ns = ((442368ull * 1000000000ull) / 6000000000ull);
		for (fragment_index=0u; fragment_index<10000u; fragment_index++)
		{
			now_ns += (2u * transfer_ns);
			require_id = 110000u + fragment_index;
			assert(SparkGlm52JitKvPoolRequireByEta(&stress_pool,now_ns,&require_id,1u,now_ns + 20000000000u) == SPARK_STATUS_OK);
		}
		assert(stress_pool.overflow_drain_count != 0u);
		assert(stress_pool.transfer_count <= SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS);
	}
	// Heap root is always the farthest-future need among residents.
	{
		uint32_t root = stress_pool.eviction_heap[0u],child;
		for (child=1u; child<stress_pool.eviction_heap_count && child<7u; child++)
			assert(stress_pool.fragments[root].next_need_ns >= stress_pool.fragments[stress_pool.eviction_heap[child]].next_need_ns);
	}
}

static void SparkTestKvDedupSharingRefcountAndClusterIntegrity(void)
{
	SparkGlm52KvDedupConfiguration configuration;
	uint32_t physical_id,shared,freed,free_physical;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_KV_DEDUP_ABI_VERSION;
	configuration.table_capacity = 1024u;
	assert(SparkGlm52KvDedupInitialize(&test_dedup,&configuration) == SPARK_STATUS_OK);
	// Non-power-of-two capacity is rejected (reuse the static instance; the
	// dedup struct is far too large to place on the stack).
	{
		SparkGlm52KvDedupConfiguration bad = configuration;
		bad.table_capacity = 1000u;
		assert(SparkGlm52KvDedupInitialize(&test_dedup,&bad) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(SparkGlm52KvDedupInitialize(&test_dedup,&configuration) == SPARK_STATUS_OK);
	}
	assert(SparkGlm52KvDedupResolve(&test_dedup,SPARK_GLM52_KV_DEDUP_EMPTY_HASH,1u,&physical_id,&shared) == SPARK_STATUS_INVALID_ARGUMENT);
	// First sight adopts the proposed physical id; repeats share it, refcount grows.
	assert(SparkGlm52KvDedupResolve(&test_dedup,0xA5A5u,100u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(physical_id == 100u && shared == 0u);
	assert(SparkGlm52KvDedupResolve(&test_dedup,0xA5A5u,200u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(physical_id == 100u && shared == 1u);
	assert(SparkGlm52KvDedupResolve(&test_dedup,0xA5A5u,300u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(physical_id == 100u && shared == 1u);
	// Distinct content gets its own physical id, never shared.
	assert(SparkGlm52KvDedupResolve(&test_dedup,0x5A5Au,400u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(physical_id == 400u && shared == 0u);
	assert(test_dedup.live_entry_count == 2u);
	// The physical fragment frees only when the last reference drops.
	assert(SparkGlm52KvDedupRelease(&test_dedup,0xA5A5u,&freed,&free_physical) == SPARK_STATUS_OK && freed == 0u);
	assert(SparkGlm52KvDedupRelease(&test_dedup,0xA5A5u,&freed,&free_physical) == SPARK_STATUS_OK && freed == 0u);
	assert(SparkGlm52KvDedupRelease(&test_dedup,0xA5A5u,&freed,&free_physical) == SPARK_STATUS_OK && freed == 1u && free_physical == 100u);
	assert(test_dedup.live_entry_count == 1u);
	// Probe-cluster integrity: three hashes to one home slot, free the middle,
	// the trailing entries must still resolve after the backward-shift reinsert.
	assert(SparkGlm52KvDedupResolve(&test_dedup,0x30u,600u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(SparkGlm52KvDedupResolve(&test_dedup,0x30u + 1024u,601u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(SparkGlm52KvDedupResolve(&test_dedup,0x30u + 2048u,602u,&physical_id,&shared) == SPARK_STATUS_OK);
	assert(SparkGlm52KvDedupRelease(&test_dedup,0x30u + 1024u,&freed,&free_physical) == SPARK_STATUS_OK && freed == 1u);
	assert(SparkGlm52KvDedupResolve(&test_dedup,0x30u + 2048u,999u,&physical_id,&shared) == SPARK_STATUS_OK && physical_id == 602u && shared == 1u);
	assert(SparkGlm52KvDedupResolve(&test_dedup,0x30u,999u,&physical_id,&shared) == SPARK_STATUS_OK && physical_id == 600u && shared == 1u);
}

static SparkGlm52JitKvPool property_pool;

static uint32_t SparkTestJitKvPoolReferenceVictim(void)
{
	uint32_t best = UINT32_MAX,fragment_index;
	uint64_t best_need = 0u;
	for (fragment_index=0u; fragment_index<property_pool.fragment_capacity; fragment_index++)
	{
		const SparkGlm52JitKvFragment *fragment = &property_pool.fragments[fragment_index];
		if ( fragment->state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM )
			continue;
		if ( best == UINT32_MAX || fragment->next_need_ns > best_need ||
			(fragment->next_need_ns == best_need && fragment_index > best) )
		{
			best_need = fragment->next_need_ns;
			best = fragment_index;
		}
	}
	return(best);
}

static void SparkTestJitKvPoolHeapMatchesReferenceUnderRandomOps(void)
{
	SparkGlm52JitKvPoolConfiguration configuration;
	uint32_t fragment_index,operation_index,random_state = 42u;
	uint64_t now_ns = 0u;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	configuration.fragment_capacity = 512u;
	configuration.dram_fragment_capacity = 64u;
	configuration.fragment_bytes = 442368u;
	configuration.nvme_bytes_per_second = 6000000000u;
	assert(SparkGlm52JitKvPoolInitialize(&property_pool,&configuration) == SPARK_STATUS_OK);
	for (fragment_index=0u; fragment_index<512u; fragment_index++)
		assert(SparkGlm52JitKvPoolAdmitFragment(&property_pool,fragment_index,fragment_index,0u,fragment_index < 64u ? SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM : SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME) == SPARK_STATUS_OK);
	for (operation_index=0u; operation_index<50000u; operation_index++)
	{
		uint32_t require_id,dram_count = 0u,staging_in = 0u,staging_out = 0u;
		uint64_t need_ns;
		SparkStatus status;
		random_state = (random_state * 1664525u + 1013904223u);
		require_id = (random_state >> 8) % 512u;
		random_state = (random_state * 1664525u + 1013904223u);
		need_ns = (now_ns + 1000000u + (uint64_t)((random_state >> 8) % 50000u) * 1000u);
		if ( property_pool.eviction_heap_count != 0u )
			assert(property_pool.eviction_heap[0u] == SparkTestJitKvPoolReferenceVictim());
		status = SparkGlm52JitKvPoolRequireByEta(&property_pool,now_ns,&require_id,1u,need_ns);
		assert(status == SPARK_STATUS_OK || status == SPARK_STATUS_CAPACITY_EXCEEDED);
		now_ns += 18432u;
		if ( (operation_index & 63u) == 0u )
			SparkGlm52JitKvPoolTick(&property_pool,now_ns);
		if ( (operation_index & 1023u) != 0u )
			continue;
		for (fragment_index=0u; fragment_index<512u; fragment_index++)
		{
			uint32_t state = property_pool.fragments[fragment_index].state;
			dram_count += (state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM ? 1u : 0u);
			staging_in += (state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN ? 1u : 0u);
			staging_out += (state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_OUT ? 1u : 0u);
		}
		assert(dram_count == property_pool.dram_resident_count);
		assert(staging_in == property_pool.staging_in_count);
		assert(staging_out == property_pool.staging_out_count);
		assert(dram_count == property_pool.eviction_heap_count);
		assert(property_pool.dram_resident_count + property_pool.staging_in_count <= configuration.dram_fragment_capacity);
	}
}

int main(void)
{
	SparkTestExpertQueueThresholdDeadlineAndOrder();
	SparkTestJitKvPoolPrefetchEvictionAndLateness();
	SparkTestBatchSequenceTableLifecycleAndThreshold();
	SparkTestJitKvPoolScaleAndBurst();
	SparkTestKvDedupSharingRefcountAndClusterIntegrity();
	SparkTestJitKvPoolHeapMatchesReferenceUnderRandomOps();
	printf("test_glm52_batch_plane PASS\n");
	return(0);
}

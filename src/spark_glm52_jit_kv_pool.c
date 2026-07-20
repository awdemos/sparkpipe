#include "sparkpipe/spark_glm52_jit_kv_pool.h"

#include <string.h>

SparkStatus SparkGlm52JitKvPoolInitialize(SparkGlm52JitKvPool *pool,const SparkGlm52JitKvPoolConfiguration *configuration)
{
	if ( pool == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_GLM52_JIT_KV_POOL_ABI_VERSION ||
		configuration->fragment_capacity == 0u ||
		configuration->fragment_capacity > SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS ||
		configuration->dram_fragment_capacity == 0u ||
		configuration->dram_fragment_capacity > configuration->fragment_capacity ||
		configuration->fragment_bytes == 0u ||
		configuration->nvme_bytes_per_second == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pool,0,sizeof(*pool));
	pool->abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	pool->fragment_capacity = configuration->fragment_capacity;
	pool->dram_fragment_capacity = configuration->dram_fragment_capacity;
	pool->fragment_bytes = configuration->fragment_bytes;
	pool->nvme_bytes_per_second = configuration->nvme_bytes_per_second;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52JitKvPoolAdmitFragment(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint64_t sequence_id,uint32_t fragment_index_in_sequence,uint32_t initial_state)
{
	SparkGlm52JitKvFragment *fragment;
	if ( pool == 0 || fragment_id >= pool->fragment_capacity ||
		(initial_state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME &&
		 initial_state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	fragment = &pool->fragments[fragment_id];
	if ( fragment->state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( initial_state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM )
	{
		if ( pool->dram_resident_count >= pool->dram_fragment_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		pool->dram_resident_count += 1u;
	}
	fragment->sequence_id = sequence_id;
	fragment->fragment_index_in_sequence = fragment_index_in_sequence;
	fragment->state = initial_state;
	fragment->next_need_ns = UINT64_MAX;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52JitKvPoolSelectEvictionVictim(const SparkGlm52JitKvPool *pool)
{
	uint64_t best_need_ns = 0u;
	uint32_t fragment_index,victim = UINT32_MAX;
	for (fragment_index=0u; fragment_index<pool->fragment_capacity; fragment_index++)
	{
		const SparkGlm52JitKvFragment *fragment = &pool->fragments[fragment_index];
		if ( fragment->state != SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM )
			continue;
		if ( victim == UINT32_MAX || fragment->next_need_ns > best_need_ns ||
			(fragment->next_need_ns == best_need_ns && fragment_index < victim) )
		{
			best_need_ns = fragment->next_need_ns;
			victim = fragment_index;
		}
	}
	return(victim);
}

static SparkStatus SparkGlm52JitKvPoolQueueTransfer(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint32_t direction_in,uint64_t now_ns)
{
	SparkGlm52JitKvTransfer *transfer;
	uint64_t start_ns,duration_ns;
	if ( pool->transfer_count >= SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	start_ns = (pool->nvme_busy_until_ns > now_ns ? pool->nvme_busy_until_ns : now_ns);
	duration_ns = ((pool->fragment_bytes * 1000000000u) / pool->nvme_bytes_per_second);
	if ( duration_ns == 0u )
		duration_ns = 1u;
	transfer = &pool->transfers[pool->transfer_count];
	transfer->fragment_id = fragment_id;
	transfer->direction_in = direction_in;
	transfer->start_ns = start_ns;
	transfer->done_ns = (start_ns + duration_ns);
	pool->transfer_count += 1u;
	pool->nvme_busy_until_ns = transfer->done_ns;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52JitKvPoolStageIn(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint64_t now_ns)
{
	SparkGlm52JitKvFragment *fragment = &pool->fragments[fragment_id];
	SparkStatus status;
	if ( pool->dram_resident_count + pool->staging_in_count >= pool->dram_fragment_capacity )
	{
		uint32_t victim = SparkGlm52JitKvPoolSelectEvictionVictim(pool);
		if ( victim == UINT32_MAX )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( pool->fragments[victim].next_need_ns <= fragment->next_need_ns )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = SparkGlm52JitKvPoolQueueTransfer(pool,victim,0u,now_ns);
		if ( status != SPARK_STATUS_OK )
			return(status);
		pool->fragments[victim].state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_OUT;
		pool->dram_resident_count -= 1u;
		pool->staging_out_count += 1u;
		pool->stage_out_count += 1u;
	}
	status = SparkGlm52JitKvPoolQueueTransfer(pool,fragment_id,1u,now_ns);
	if ( status != SPARK_STATUS_OK )
		return(status);
	fragment->state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN;
	pool->staging_in_count += 1u;
	pool->stage_in_count += 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52JitKvPoolRequireByEta(SparkGlm52JitKvPool *pool,uint64_t now_ns,const uint32_t *fragment_ids,uint32_t fragment_count,uint64_t need_ns)
{
	SparkGlm52JitKvFragment *fragment;
	uint32_t request_index;
	SparkStatus status;
	if ( pool == 0 || (fragment_ids == 0 && fragment_count != 0u) || need_ns < now_ns )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (request_index=0u; request_index<fragment_count; request_index++)
	{
		if ( fragment_ids[request_index] >= pool->fragment_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		fragment = &pool->fragments[fragment_ids[request_index]];
		if ( fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE )
			return(SPARK_STATUS_NOT_FOUND);
		if ( need_ns < fragment->next_need_ns )
			fragment->next_need_ns = need_ns;
		if ( fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM ||
			fragment->state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN )
		{
			pool->hit_count += 1u;
			continue;
		}
		pool->miss_count += 1u;
		status = SparkGlm52JitKvPoolStageIn(pool,fragment_ids[request_index],now_ns);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52JitKvPoolTick(SparkGlm52JitKvPool *pool,uint64_t now_ns)
{
	SparkGlm52JitKvFragment *fragment;
	uint32_t transfer_index,kept = 0u;
	if ( pool == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (transfer_index=0u; transfer_index<pool->transfer_count; transfer_index++)
	{
		SparkGlm52JitKvTransfer *transfer = &pool->transfers[transfer_index];
		if ( transfer->done_ns > now_ns )
		{
			pool->transfers[kept] = *transfer;
			kept += 1u;
			continue;
		}
		fragment = &pool->fragments[transfer->fragment_id];
		if ( transfer->direction_in != 0u )
		{
			fragment->state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM;
			pool->staging_in_count -= 1u;
			pool->dram_resident_count += 1u;
			if ( transfer->done_ns > fragment->next_need_ns )
				pool->late_count += 1u;
		}
		else
		{
			fragment->state = SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME;
			fragment->next_need_ns = UINT64_MAX;
			pool->staging_out_count -= 1u;
		}
	}
	pool->transfer_count = kept;
	return(SPARK_STATUS_OK);
}

uint32_t SparkGlm52JitKvPoolFragmentIsResident(const SparkGlm52JitKvPool *pool,uint32_t fragment_id)
{
	if ( pool == 0 || fragment_id >= pool->fragment_capacity )
		return(0u);
	return(pool->fragments[fragment_id].state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM ? 1u : 0u);
}

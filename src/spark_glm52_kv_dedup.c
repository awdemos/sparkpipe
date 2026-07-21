#include "sparkpipe/spark_glm52_kv_dedup.h"

#include <stddef.h>
#include <string.h>

static uint32_t SparkGlm52KvDedupIsPowerOfTwo(uint32_t value) { return(value != 0u && (value & (value - 1u)) == 0u); }

static uint64_t SparkGlm52KvDedupMix(uint64_t hash)
{
	hash ^= (hash >> 33);
	hash *= 0xFF51AFD7ED558CCDu;
	hash ^= (hash >> 33);
	return(hash);
}

SparkStatus SparkGlm52KvDedupInitialize(SparkGlm52KvDedup *dedup,const SparkGlm52KvDedupConfiguration *configuration)
{
	if ( dedup == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_GLM52_KV_DEDUP_ABI_VERSION ||
		configuration->table_capacity == 0u ||
		configuration->table_capacity > SPARK_GLM52_KV_DEDUP_MAX_ENTRIES ||
		!SparkGlm52KvDedupIsPowerOfTwo(configuration->table_capacity) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(dedup,0,offsetof(SparkGlm52KvDedup,entries));
	memset(dedup->entries,0,sizeof(SparkGlm52KvDedupEntry) * configuration->table_capacity);
	dedup->abi_version = SPARK_GLM52_KV_DEDUP_ABI_VERSION;
	dedup->table_capacity = configuration->table_capacity;
	dedup->table_mask = (configuration->table_capacity - 1u);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52KvDedupFindSlot(SparkGlm52KvDedup *dedup,uint64_t content_hash,uint32_t *found_out)
{
	uint32_t slot = (uint32_t)(SparkGlm52KvDedupMix(content_hash) & dedup->table_mask),probes = 0u;
	*found_out = 0u;
	while (probes <= dedup->table_mask)
	{
		const SparkGlm52KvDedupEntry *entry = &dedup->entries[slot];
		if ( entry->content_hash == SPARK_GLM52_KV_DEDUP_EMPTY_HASH )
		{
			if ( probes > dedup->max_probe_length )
				dedup->max_probe_length = probes;
			return(slot);
		}
		if ( entry->content_hash == content_hash )
		{
			if ( probes > dedup->max_probe_length )
				dedup->max_probe_length = probes;
			*found_out = 1u;
			return(slot);
		}
		slot = ((slot + 1u) & dedup->table_mask);
		probes += 1u;
	}
	return(UINT32_MAX);
}

SparkStatus SparkGlm52KvDedupResolve(SparkGlm52KvDedup *dedup,uint64_t content_hash,uint32_t proposed_physical_id,uint32_t *physical_id_out,uint32_t *shared_out)
{
	SparkGlm52KvDedupEntry *entry;
	uint32_t slot,found;
	if ( dedup == 0 || physical_id_out == 0 || shared_out == 0 ||
		content_hash == SPARK_GLM52_KV_DEDUP_EMPTY_HASH )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = SparkGlm52KvDedupFindSlot(dedup,content_hash,&found);
	if ( slot == UINT32_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	entry = &dedup->entries[slot];
	if ( found != 0u )
	{
		entry->reference_count += 1u;
		dedup->resolve_hit_count += 1u;
		dedup->shared_reference_count += 1u;
		*physical_id_out = entry->physical_fragment_id;
		*shared_out = 1u;
		return(SPARK_STATUS_OK);
	}
	entry->content_hash = content_hash;
	entry->physical_fragment_id = proposed_physical_id;
	entry->reference_count = 1u;
	dedup->live_entry_count += 1u;
	dedup->resolve_insert_count += 1u;
	*physical_id_out = proposed_physical_id;
	*shared_out = 0u;
	return(SPARK_STATUS_OK);
}

static void SparkGlm52KvDedupReinsertCluster(SparkGlm52KvDedup *dedup,uint32_t start_slot)
{
	uint32_t slot = ((start_slot + 1u) & dedup->table_mask);
	while (dedup->entries[slot].content_hash != SPARK_GLM52_KV_DEDUP_EMPTY_HASH)
	{
		SparkGlm52KvDedupEntry moved = dedup->entries[slot];
		uint32_t found,target;
		dedup->entries[slot].content_hash = SPARK_GLM52_KV_DEDUP_EMPTY_HASH;
		target = SparkGlm52KvDedupFindSlot(dedup,moved.content_hash,&found);
		dedup->entries[target] = moved;
		slot = ((slot + 1u) & dedup->table_mask);
	}
}

SparkStatus SparkGlm52KvDedupRelease(SparkGlm52KvDedup *dedup,uint64_t content_hash,uint32_t *free_out,uint32_t *physical_id_out)
{
	SparkGlm52KvDedupEntry *entry;
	uint32_t slot,found;
	if ( dedup == 0 || free_out == 0 || physical_id_out == 0 ||
		content_hash == SPARK_GLM52_KV_DEDUP_EMPTY_HASH )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = SparkGlm52KvDedupFindSlot(dedup,content_hash,&found);
	if ( slot == UINT32_MAX || found == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	entry = &dedup->entries[slot];
	entry->reference_count -= 1u;
	*free_out = 0u;
	*physical_id_out = entry->physical_fragment_id;
	if ( entry->reference_count == 0u )
	{
		entry->content_hash = SPARK_GLM52_KV_DEDUP_EMPTY_HASH;
		dedup->live_entry_count -= 1u;
		dedup->release_free_count += 1u;
		SparkGlm52KvDedupReinsertCluster(dedup,slot);
		*free_out = 1u;
	}
	else
		dedup->shared_reference_count -= 1u;
	return(SPARK_STATUS_OK);
}

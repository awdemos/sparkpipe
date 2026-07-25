// Shared-prefix admission: the composition production must use.
//
// SparkGlm52KvDedup owns fragment IDENTITY - it maps a content hash to one
// physical fragment id and refcounts it. SparkGlm52JitKvPool owns fragment
// RESIDENCY - which physical fragments are in DRAM and when they page. They
// are separate concerns and neither should reimplement the other; the caller
// composes them, resolving identity first and admitting to the pool only when
// dedup reports the fragment is new.
//
// This is what a B8 shared-prefix chat batch must do: eight rows presenting
// the same prefix must produce ONE physical fragment and ONE pool admission,
// not eight. Both components already exist and are tested individually; this
// pins the composition so the production call site has a reference.
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "sparkpipe/spark_glm52_kv_dedup.h"
#include "sparkpipe/spark_glm52_jit_kv_pool.h"

#define SPARK_TEST_SHARED_ROWS 8u
#define SPARK_TEST_PREFIX_HASH 0xABCDEF0123456789ull

static SparkGlm52KvDedup dedup;
static SparkGlm52JitKvPool pool;

// One row acquiring one logical block: resolve identity, and only physically
// admit when this row is the first to present that content.
static uint32_t SparkTestAcquireBlock(uint64_t content_hash, uint32_t proposed_id, uint32_t *admitted_out)
{
	uint32_t physical_id, shared;
	assert(SparkGlm52KvDedupResolve(&dedup, content_hash, proposed_id, &physical_id, &shared) == SPARK_STATUS_OK);
	*admitted_out = 0u;
	if (shared == 0u)
	{
		assert(SparkGlm52JitKvPoolAdmitFragment(&pool, physical_id, 0u, 0u, SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM) == SPARK_STATUS_OK);
		*admitted_out = 1u;
	}
	return physical_id;
}

int main(void)
{
	SparkGlm52KvDedupConfiguration dedup_configuration;
	SparkGlm52JitKvPoolConfiguration pool_configuration;
	uint32_t physical_id[SPARK_TEST_SHARED_ROWS], admitted, admissions = 0u, row, free_it, freed_id;

	memset(&dedup_configuration, 0, sizeof(dedup_configuration));
	dedup_configuration.abi_version = SPARK_GLM52_KV_DEDUP_ABI_VERSION;
	dedup_configuration.table_capacity = 1024u;
	assert(SparkGlm52KvDedupInitialize(&dedup, &dedup_configuration) == SPARK_STATUS_OK);

	memset(&pool_configuration, 0, sizeof(pool_configuration));
	pool_configuration.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	pool_configuration.fragment_capacity = 256u;
	pool_configuration.dram_fragment_capacity = 128u;
	pool_configuration.fragment_bytes = 1152u * 64u;
	pool_configuration.nvme_bytes_per_second = 6000000000ull;
	assert(SparkGlm52JitKvPoolInitialize(&pool, &pool_configuration) == SPARK_STATUS_OK);

	for (row = 0u; row < SPARK_TEST_SHARED_ROWS; row++)
	{
		physical_id[row] = SparkTestAcquireBlock(SPARK_TEST_PREFIX_HASH, 10u + row, &admitted);
		admissions += admitted;
	}
	printf("B%u shared prefix: %u physical admission(s), dedup inserts=%llu hits=%llu\n",
		SPARK_TEST_SHARED_ROWS, admissions,
		(unsigned long long)dedup.resolve_insert_count,
		(unsigned long long)dedup.resolve_hit_count);
	assert(admissions == 1u);
	assert(dedup.resolve_insert_count == 1u);
	assert(dedup.resolve_hit_count == SPARK_TEST_SHARED_ROWS - 1u);
	for (row = 1u; row < SPARK_TEST_SHARED_ROWS; row++)
		assert(physical_id[row] == physical_id[0]);
	assert(pool.dram_resident_count == 1u);
	printf("  all %u rows -> physical fragment %u, dram_resident=%u (not %u)\n",
		SPARK_TEST_SHARED_ROWS, physical_id[0], pool.dram_resident_count, SPARK_TEST_SHARED_ROWS);

	for (row = 0u; row < SPARK_TEST_SHARED_ROWS - 1u; row++)
	{
		assert(SparkGlm52KvDedupRelease(&dedup, SPARK_TEST_PREFIX_HASH, &free_it, &freed_id) == SPARK_STATUS_OK);
		assert(free_it == 0u);
	}
	printf("  %u releases: fragment still referenced\n", SPARK_TEST_SHARED_ROWS - 1u);
	assert(SparkGlm52KvDedupRelease(&dedup, SPARK_TEST_PREFIX_HASH, &free_it, &freed_id) == SPARK_STATUS_OK);
	assert(free_it == 1u && freed_id == physical_id[0]);
	printf("  final release: frees physical fragment %u\n", freed_id);

	printf("\nPASS - %u rows share one fragment; identity and residency stay separate\n", SPARK_TEST_SHARED_ROWS);
	return 0;
}

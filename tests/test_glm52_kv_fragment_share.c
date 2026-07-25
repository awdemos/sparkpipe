#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "sparkpipe/spark_glm52_jit_kv_pool.h"

static SparkGlm52JitKvPool pool;

int main(void)
{
	SparkGlm52JitKvPoolConfiguration cfg;
	uint32_t id[8],created[8],i,first;
	SparkStatus st;
	memset(&cfg,0,sizeof(cfg));
	cfg.abi_version = SPARK_GLM52_JIT_KV_POOL_ABI_VERSION;
	cfg.fragment_capacity = 256u;
	cfg.dram_fragment_capacity = 128u;
	cfg.fragment_bytes = 1152u * 64u;
	cfg.nvme_bytes_per_second = 6000000000ull;
	st = SparkGlm52JitKvPoolInitialize(&pool,&cfg);
	assert(st == SPARK_STATUS_OK);
	printf("hash_slots=%u (want >= 2*capacity=512)\n",pool.hash_slots);
	assert(pool.hash_slots >= 512u);

	/* B8 shared prefix: eight rows acquire the SAME content hash */
	for (i = 0; i < 8u; i++)
	{
		st = SparkGlm52JitKvPoolAcquireShared(&pool,0xABCDEF0123456789ull,10u + i,0u,
			SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM,&id[i],&created[i]);
		assert(st == SPARK_STATUS_OK);
	}
	first = id[0];
	printf("B8 shared prefix: admits=%llu hits=%llu\n",
		(unsigned long long)pool.share_admit_count,(unsigned long long)pool.share_hit_count);
	assert(pool.share_admit_count == 1u);
	assert(pool.share_hit_count == 7u);
	for (i = 0; i < 8u; i++)
		assert(id[i] == first);
	assert(created[0] == 1u);
	for (i = 1; i < 8u; i++)
		assert(created[i] == 0u);
	printf("  all 8 rows -> fragment %u, dram_resident=%u (want 1, not 8)\n",
		first,pool.dram_resident_count);
	assert(pool.dram_resident_count == 1u);
	assert(pool.fragments[first].reference_count == 8u);

	/* seven releases must NOT free it */
	for (i = 0; i < 7u; i++)
	{
		st = SparkGlm52JitKvPoolReleaseFragment(&pool,first);
		assert(st == SPARK_STATUS_OK);
	}
	assert(pool.fragments[first].state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM);
	assert(pool.fragments[first].reference_count == 1u);
	printf("  after 7 releases: still resident, refcount=1\n");

	/* the last release frees and unindexes */
	st = SparkGlm52JitKvPoolReleaseFragment(&pool,first);
	assert(st == SPARK_STATUS_OK);
	assert(pool.fragments[first].state == SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE);
	assert(pool.dram_resident_count == 0u);
	printf("  after 8th release: freed, dram_resident=0\n");

	/* re-acquire after free must ADMIT, not hit a stale index entry */
	st = SparkGlm52JitKvPoolAcquireShared(&pool,0xABCDEF0123456789ull,20u,0u,
		SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM,&id[0],&created[0]);
	assert(st == SPARK_STATUS_OK);
	assert(created[0] == 1u);
	printf("  re-acquire after free: created=1 (no stale hit)\n");

	/* distinct hashes must not collide into one fragment */
	st = SparkGlm52JitKvPoolAcquireShared(&pool,0x1111111111111111ull,21u,0u,
		SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM,&id[1],&created[1]);
	assert(st == SPARK_STATUS_OK && created[1] == 1u && id[1] != id[0]);
	printf("  distinct hash -> distinct fragment\n");

	/* double release past zero is refused, not silent corruption */
	SparkGlm52JitKvPoolReleaseFragment(&pool,id[1]);
	st = SparkGlm52JitKvPoolReleaseFragment(&pool,id[1]);
	assert(st == SPARK_STATUS_INVALID_ARGUMENT);
	printf("  double-release refused\n");

	printf("\nALL PASS - 8x prefix dedup verified, 1 fragment instead of 8\n");
	return 0;
}

#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

// Mooncake-style JIT KV fragment pool for the expert-queue batch plane.
// Host-side residency and transfer scheduling only; DMA is performed by the
// caller against the plans this pool emits. Deterministic: identical call
// sequences produce identical plans. Fragments are 64-token latent-KV blocks
// for the rank's own pipeline layers; the pool tiers them DRAM <-> NVMe and
// schedules stage-ins against each request wave's known arrival time, so the
// batch plane's long transit becomes prefetch lead and NVMe latency is hidden
// behind bandwidth budgeting.

#define SPARK_GLM52_JIT_KV_POOL_ABI_VERSION 2u
#define SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS 262144u
#define SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS 4096u
#define SPARK_GLM52_JIT_KV_POOL_FRAGMENT_TOKENS 64u

#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_FREE 0u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_NVME 1u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_IN 2u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_DRAM 3u
#define SPARK_GLM52_JIT_KV_FRAGMENT_STATE_STAGING_OUT 4u

/*
 * Shared-fragment index. A fragment is keyed by a CONTENT hash - the hash of
 * the token run that produced it - rather than by the sequence that happened
 * to request it first, so identical prefixes across concurrent sequences
 * resolve to one fragment. The chat path submits shared-prefix batches, where
 * the prefix would otherwise be stored and read once per row.
 *
 * The index is open-addressed with linear probing and a tombstone, sized to
 * the next power of two at or above twice the fragment capacity, so lookup is
 * O(1) rather than a scan over the fragment array. Four times the maximum
 * fragment count always covers that rounding.
 */
#define SPARK_GLM52_JIT_KV_POOL_HASH_SLOTS (SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS * 4u)
#define SPARK_GLM52_JIT_KV_POOL_HASH_TOMBSTONE 0xffffffffu

typedef struct SparkGlm52JitKvFragment
{
	uint64_t sequence_id;
	uint64_t content_hash;
	uint32_t reference_count;
	uint64_t next_need_ns;
	uint32_t state;
	uint32_t fragment_index_in_sequence;
	uint32_t heap_position;
} SparkGlm52JitKvFragment;

typedef struct SparkGlm52JitKvTransfer
{
	uint32_t fragment_id;
	uint32_t direction_in;
	uint64_t start_ns;
	uint64_t done_ns;
} SparkGlm52JitKvTransfer;

typedef struct SparkGlm52JitKvPoolConfiguration
{
	uint32_t abi_version;
	uint32_t fragment_capacity;
	uint32_t dram_fragment_capacity;
	uint64_t fragment_bytes;
	uint64_t nvme_bytes_per_second;
} SparkGlm52JitKvPoolConfiguration;

typedef struct SparkGlm52JitKvPool
{
	uint32_t abi_version;
	uint32_t fragment_capacity;
	uint32_t dram_fragment_capacity;
	uint32_t dram_resident_count;
	uint32_t staging_in_count;
	uint32_t staging_out_count;
	uint64_t fragment_bytes;
	uint64_t nvme_bytes_per_second;
	uint64_t nvme_busy_until_ns;
	uint64_t stage_in_count;
	uint64_t stage_out_count;
	uint64_t hit_count;
	uint64_t miss_count;
	uint64_t late_count;
	uint64_t overflow_drain_count;
	uint32_t eviction_heap_count;
	uint32_t transfer_head;
	uint32_t transfer_count;
	uint32_t hash_slots;
	uint64_t share_hit_count;
	uint64_t share_admit_count;
	SparkGlm52JitKvFragment fragments[SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS];
	SparkGlm52JitKvTransfer transfers[SPARK_GLM52_JIT_KV_POOL_MAX_PENDING_TRANSFERS];
	uint32_t eviction_heap[SPARK_GLM52_JIT_KV_POOL_MAX_FRAGMENTS];
	uint32_t hash_table[SPARK_GLM52_JIT_KV_POOL_HASH_SLOTS];
} SparkGlm52JitKvPool;

SparkStatus SparkGlm52JitKvPoolInitialize(SparkGlm52JitKvPool *pool,const SparkGlm52JitKvPoolConfiguration *configuration);
SparkStatus SparkGlm52JitKvPoolAdmitFragment(SparkGlm52JitKvPool *pool,uint32_t fragment_id,uint64_t sequence_id,uint32_t fragment_index_in_sequence,uint32_t initial_state);
SparkStatus SparkGlm52JitKvPoolRequireByEta(SparkGlm52JitKvPool *pool,uint64_t now_ns,const uint32_t *fragment_ids,uint32_t fragment_count,uint64_t need_ns);
SparkStatus SparkGlm52JitKvPoolTick(SparkGlm52JitKvPool *pool,uint64_t now_ns);
uint32_t SparkGlm52JitKvPoolFragmentIsResident(const SparkGlm52JitKvPool *pool,uint32_t fragment_id);
SparkStatus SparkGlm52JitKvPoolAcquireShared(SparkGlm52JitKvPool *pool,uint64_t content_hash,uint32_t candidate_fragment_id,uint32_t fragment_index_in_sequence,uint32_t initial_state,uint32_t *fragment_id_out,uint32_t *created_out);
SparkStatus SparkGlm52JitKvPoolReleaseFragment(SparkGlm52JitKvPool *pool,uint32_t fragment_id);

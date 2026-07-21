#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

// Content-addressed KV fragment dedup. Sits above the physical JIT pool: maps a
// fragment content hash to a single physical fragment id, refcounted, so that
// byte-identical prefix fragments shared across sequences (system prompt, tool
// schemas, few-shot examples, shared longmem memory) resolve to one physical
// copy instead of one per sequence. Strictly never worse than per-sequence
// storage: a fragment with no content match gets a fresh physical id exactly as
// before, while a match returns the existing id and increments its refcount.
// The physical fragment is releasable only when its refcount reaches zero.
//
// Open-addressed hash table, linear probing, power-of-two capacity, no dynamic
// allocation. Determinism: identical admit sequences produce identical physical
// id assignments for the ring SHA gates. Sizing contract: provision at least
// twice the expected distinct-fragment count. Linear probing degrades sharply
// past a 0.7 load factor; max_probe_length is exported so a caller can detect a
// table that is running too full before resolve latency becomes a problem.

#define SPARK_GLM52_KV_DEDUP_ABI_VERSION 1u
#define SPARK_GLM52_KV_DEDUP_MAX_ENTRIES 524288u
#define SPARK_GLM52_KV_DEDUP_EMPTY_HASH 0u

typedef struct SparkGlm52KvDedupEntry
{
	uint64_t content_hash;
	uint32_t physical_fragment_id;
	uint32_t reference_count;
} SparkGlm52KvDedupEntry;

typedef struct SparkGlm52KvDedupConfiguration
{
	uint32_t abi_version;
	uint32_t table_capacity;
} SparkGlm52KvDedupConfiguration;

typedef struct SparkGlm52KvDedup
{
	uint32_t abi_version;
	uint32_t table_capacity;
	uint32_t table_mask;
	uint32_t live_entry_count;
	uint64_t resolve_hit_count;
	uint64_t resolve_insert_count;
	uint64_t shared_reference_count;
	uint64_t release_free_count;
	uint32_t max_probe_length;
	SparkGlm52KvDedupEntry entries[SPARK_GLM52_KV_DEDUP_MAX_ENTRIES];
} SparkGlm52KvDedup;

SparkStatus SparkGlm52KvDedupInitialize(SparkGlm52KvDedup *dedup,const SparkGlm52KvDedupConfiguration *configuration);
// Resolve a content hash to a physical id. On first sight of a hash the caller's
// proposed_physical_id is adopted and returned (a fresh physical fragment); on a
// repeat the stored id is returned and shared_out is set, and the caller must
// not physically store its proposed id. reference_count is incremented either way.
SparkStatus SparkGlm52KvDedupResolve(SparkGlm52KvDedup *dedup,uint64_t content_hash,uint32_t proposed_physical_id,uint32_t *physical_id_out,uint32_t *shared_out);
// Drop one reference. free_out is set when the last reference is released and the
// physical id may be returned to the pool.
SparkStatus SparkGlm52KvDedupRelease(SparkGlm52KvDedup *dedup,uint64_t content_hash,uint32_t *free_out,uint32_t *physical_id_out);

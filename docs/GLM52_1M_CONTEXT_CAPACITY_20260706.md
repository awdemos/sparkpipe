# 1M Context: Capacity Layer, Corrected Map, and the Two Remaining Projects - 2026-07-06

## Retraction and corrected architecture map

My audit claimed no full-context latent or index-key pool existed. Wrong at
the architecture level: spark_glm52_kv_cache.h defines a physical block
arena (ALLOCATED/RESIDENT flags, host-side block indices, per-lane counts),
the DSA kernels (DsaScoreKernel, DsaSelectRadixTopk, SelectedBlockBuild,
KvFragmentPrefetch/Save) resolve slots through block tables against
kv_block_count/cache_token_capacity, and the service derives its block pool
from a context-token constant. The paged tier exists. What pinned the
system to a 2048-token window and 64k contexts was CONFIGURATION: the
builder's 32-blocks-per-sequence constant, sequence-major cache sizing, a
selected-count-sized key_index_cache, and the 65536 service constant.

That last one was also a latent bug class: DsaScoreKernel addresses
key_index_cache by resolved pool slot, but the builder sized the buffer at
b x SELECTED_TOKEN_COUNT - consistent only while capacity equals the
window. Any capacity raise without this PR's reshape would have been
out-of-bounds.

## What this PR changes (verified mechanical)

One shared capacity source in spark_glm52_kv_cache.h:
SPARK_GLM52_KV_CONTEXT_TOKENS = 1048576 (per-sequence context ceiling) and
SPARK_GLM52_KV_POOL_TOKENS = 4194304 (total pooled KV tokens per stage),
with a preprocessor divisibility check. Consumers derive:

- builder: blocks-per-sequence = CONTEXT/64 = 16384; mla_cache and
  key_index_cache become pool-shaped (POOL_TOKENS x 576 and
  POOL_TOKENS x 128), slot-addressed as the kernels already assume;
  node cache_token_capacity/kv_block_count = pool totals; block tables
  b x 16384. The JIT selected-set buffers (key_nope/value at
  b x 2048 x head-dims) are correct as-is and untouched.
- service backend: CONTEXT_TOKENS aliases the shared constant, which
  scales its derived block pool and prefix bindings.

Per-stage memory at these defaults: mla pool 27.6 GB (4M x 1152 B x 6
layers), index-key pool 6.1 GB, block tables 64 MB at B1024, JIT buffers
~2 GB, request token storage 512 MB host. With ~60 GB FP8 weights: ~96 GB
of 128 GB committed. Capacity: 4 concurrent 1M-context users, or ~62 at
64k, or mixes - the pool is the product knob. B4/B64/B1024 share the same
footprint; batch size costs nothing extra, exactly as requested.

## REQUIRED on spark2 (the growth path is the unverified piece)

1. A real >2048-token decode: prefill ~8k tokens, decode, verify the model
   attends early context (needle retrieval). This exercises the service
   block allocator growing lanes past 32 blocks (lane_capacity handling,
   KvFragmentPrefetch behavior at depth) - the one path I could not
   execute from here.
2. Verify dsa_candidate_count derives from context_length (not a 2048
   constant) at the score/select call sites during that run.
3. Full make test + the existing token-match gates (allocation reshape
   should be bit-neutral at old workloads; the 27.6 GB pool memset at
   startup is one-time).

## Project 1: DSA-sparse prefill (kills the N^2 that matters)

Full-attention chunked prefill is O(N^2) at 512-dim x 64 heads: ~27 PFLOP
at 64k, unusable at 1M. DSA-sparse prefill is O(N^2) only in the INDEXER
(128-dim single-vector dots: ~256 TFLOP at 1M, tens of seconds once) plus
O(N x 2048) attention - two orders of magnitude off the wall.

The trap that forbids the naive kernel: per-query selected attention with
no KV sharing reads 2048 x 1152 B per query = ~14 TB/stage at 1M. Real
implementations exploit selection overlap between adjacent queries. Two
candidate strategies, both reusing the existing score/radix/build kernels:
(a) tile-union - 16-query tiles attend the union of their selections
through the WMMA prefill kernel with a gathered slot list; (b)
chunk-shared selection - one selection per chunk prefix plus the causal
in-chunk window. Both are approximations of per-query DSA; the grounding
prerequisite is the reference semantics (upstream DSA prefill behavior)
and an accuracy gate (long-context needle fixtures) before either ships
to a public product. This is the next kernel project and it is mine.

## Project 2: dspark serving integration (and the MTP question)

Grounded: the serving stack has zero dspark references; topology sidebands
exist only for DSA index sharing. The five tap layers live on five
different ranks, so integration = a new sideband kind exporting tap hidden
states to the final rank (the sideband/transport machinery is the
template), the backend's TapOutputPointers/StageLane wiring in the
adapter, and a Draft -> MTP-accept loop in the serving engine - the MTP
draft/accept/commit buffers already exist in the builder. Order of work:
first validate MTP itself end-to-end (draft acceptance rate > 0 and
committed tokens match greedy decode - the doubt is warranted until
measured), then the tap sideband, then Draft v1 per-lane, then batched
Draft v2. Speculation is accuracy-neutral by construction (rejected
drafts fall back to verifier tokens), which makes it the safest large
multiplier: 2-3x decode on top of everything above.

## Priority order

1. This PR's spark2 verification (the >2048 decode test).
2. DSA-sparse prefill (TTFT is the public-site killer at any long context).
3. MTP validation, then dspark sideband + draft loop.
4. Absorbed attention v2 (64 heads/block, dynamic smem): 4x fewer latent
   bytes in the now-default decode path.
5. Web-side work resumes after 1-3, per the stated priority.

# Remaining Work After Phase 4

Phase 4 closes the host-verifiable shared CUDA contracts and model precision
seams that would invalidate later measurements. The items below remain
unqualified or unfinished.

## P0: exact CUDA qualification

1. Install CUDA 13.3.x in Linux with `nvcc`, `ptxas`, `nvlink`, `cuobjdump`,
   CUDA headers, cuBLAS/cuBLASLt, and the required verbs development headers.
2. Run `tools/cuda13_sm121a_compile_gate.sh` with exact `compute_121a` PTX
   generation and exact `sm_121a` object and device-link generation.
3. Repair every CUDA compile, template-instantiation, device-link,
   architecture-instruction, register-spill, local-memory, and shared-memory
   failure exposed by that gate.
4. Retain independent BF16, FP8 E4M3, MXFP4 E2M1, UE4M3, and UE8M0 numerical
   vectors, including multiple rows, experts, output tiles, and K-scale groups.
5. Run Compute Sanitizer memory, race, synchronization, and initialization
   checks on DGX Spark.
6. Retain exact-package latency, throughput, occupancy, memory-bandwidth, and
   power receipts for every supported batch/context/topology bucket.

## P0: Kimi K3 execution closure

1. Connect the K3 resident-stage doorway to the shipping stage executor.
2. Carry request generation, step generation, exact KDA replay slabs, accepted
   prefix length, bonus-token ownership, and committed-state ownership through
   the distributed transaction protocol.
3. Integrate the versioned pipeline sideband so only newly produced Block
   AttnRes block state and the live partial cross a stage boundary; never resend
   the complete bank at every boundary.
4. Link KDA, Gated MLA, Block AttnRes, Stable LatentMoE, two full-width shared
   experts, and the BF16-activation/MXFP4-weight routed path into one package.
5. Generate all model geometry from `model_contracts/k3_authoritative.json` and
   remove the remaining hand-maintained duplicate constants.
6. Add checkpoint-derived vectors for KDA decode, exact accepted-prefix replay,
   MLA, AttnRes, SiTU-GLU, frozen-bias routing, shared experts, routed experts,
   and complete layers.
7. Wire and qualify the EAGLE-3-style draft model, including low/mid/high
   AttnRes feature fusion and seven-step recurrent drafting.
8. Qualify the requested BF16 expert-activation variant independently. The
   upstream report's native post-training deployment uses MXFP4 expert weights
   with MXFP8 expert activations, not BF16 expert activations.

## P0: GLM 5.2 execution closure

1. Collapse the parallel historical and first-party fragments into one
   link-complete production module.
2. Remove every absent or deprecated CUDA source and W8LUT/NVFP4-era build seam
   from the FP8-expert package.
3. Connect bulk prefill, decode, MTP, and DSpark to the same current first-party
   entry points and exact precision contract.
4. Construct route metadata on device and feed complete scheduler layer batches
   into sealed expert queues.
5. Preserve one active-expert weight load and one grouped GEMM firing per expert
   per sealed layer batch, with weight-stationary continuation for experts that
   require more than one M tile.
6. Validate the FP8 expert pack against exact checkpoint metadata and retain
   tensor hashes, scale geometry, and numerical receipts.
7. Verify that attention, dense/shared FFN, routers, latent projections, caches,
   and final head remain BF16 while only routed expert weights are FP8 E4M3.

## P0: Qwen 3.6 27B execution closure

1. Connect restored work control to the shipping stage and queue protocol.
2. Verify BF16 through pack, binding, attention, GDN, dense FFN, KV/state cache,
   prefix restore, final head, and output streaming.
3. Add checkpoint-derived numerical vectors and complete stage receipts.

## P0: DeepSeek V4 Flash and Pro execution closure

1. Implement a complete Flash resident stage consuming the class-exact sliding,
   compressed-history, and compressor/indexer arenas.
2. Implement an independent Pro resident stage; never dispatch Pro through
   Flash layer schedules or cache geometry.
3. Implement every generated sliding-attention, CSA, and HCA layer class.
4. Implement the four-stream manifold-constrained hyper-connection state and
   hash-routed bootstrap layers.
5. Complete grouped low-rank output projection, shared expert, routed expert,
   and MTP paths.
6. Validate checkpoint FP4 decoding and FP8 block-scale geometry with real
   weights for both variants.
7. Add cache page reference counting, prefix restoration, eviction, transfer,
   and stale-generation tests.

## P0: distributed queue correctness

1. Define one prepare, accept, execute, and commit transaction for decode,
   prefill, speculation, cancellation, and release across every rank.
2. Put request generation and step generation in every work and completion
   packet.
3. Require downstream acceptance before destructive local state changes and
   define the rollback/cancel action for every failure point.
4. Add an acknowledged commit boundary, reconnect deduplication, and idempotent
   replay so a partially transmitted packet cannot execute twice.
5. Separate transport-window credit, resident admission reservation, execution
   occupancy, and completion ownership. They must not share one counter.
6. Remove duplicate credit returns and make all completion queues lossless or
   deterministically fail the owning request under saturation.
7. Validate release and cancellation by request generation, step generation,
   and current owner; coalesce duplicates.
8. Remove synchronous resident-result waits and head-of-line stalls from rank
   and backend event loops.
9. Drain completion backlogs according to measured backlog and elapsed time,
   not a fixed iteration budget.
10. Replace normal-path host `cudaStreamSynchronize` calls with stream-ordered
    completion records and asynchronous token delivery.

## P1: grouped-MoE production performance

1. Create expert-major descriptors entirely on device without a host
   synchronization.
2. Retain one expert-weight load and one grouped GEMM firing per active expert
   per sealed layer batch.
3. Add a weight-stationary multi-tile continuation when one expert exceeds the
   selected M tile.
4. Overlap shared experts, route construction, expert weight streaming,
   inter-stage transport, and route folding.
5. Retain real route-skew histograms and replace uniform-routing estimates.
6. Keep low-latency and sealed-throughput policies as separately qualified
   package modes.
7. Use token-centric/warp-decode kernels for small expert rows and grouped
   tensor-core kernels only after the measured crossover.

## P1: single-rail transport and topology

1. Bring up the direct physical ring with pipeline stage order matching physical
   adjacency and explicit clockwise/counter-clockwise routing.
2. Bring up one MikroTik 804 as one switched 100 Gbit/s rail after ring
   correctness is proven.
3. Pre-advertise persistent RDMA receive slots instead of sending a
   `RECEIVE_READY` control message for every payload.
4. Add slot generations so delayed writes cannot target a reused buffer.
5. Add bounded MR-cache eviction. The current cache records a last-use epoch but
   still returns `CAPACITY_EXCEEDED` when every slot has been populated.
6. Poll completion queues in batches rather than one work completion per call.
7. Add an autonomous progress worker so network progress is not dependent on
   request-thread polling.
8. Validate mapped-pinned and device-direct boundary buffers on the actual NIC,
   CUDA driver, peer-memory support, and IOMMU configuration.
9. Rotate independent B1 boundary slots across QPs without reordering packets
   within one request.
10. Measure PP8, PP13, and PP16 from B1 through B1024 and select topology by
    retained receipt rather than a global rule.
11. Keep the second switch and dual-rail scheduling disabled until every
    single-rail ownership, ordering, failure, and retry invariant is proven.

## P1: architecture and DRY closure

1. Move GLM-specific scheduler, ring, cache, stage, request, and speculation
   contracts out of neutral-looking public headers under `include/sparkpipe/`.
2. Replace generic names such as `spark_resident_decode_stage.h` and
   `spark_ring_runtime.h` where their fields and constants are GLM-specific.
3. Generate K3 and DSV4 configuration surfaces from their authoritative
   contracts without duplicate hand-written geometry.
4. Keep stable semantic primitives shared, but keep exact model schedules,
   formats, fusions, and transport policy inside model-family firmware.

## Newly exposed or corrected in Phase 4

1. The old architecture target referenced a missing core-boundary auditor and
   nonexistent per-module Makefile targets. The executable audit and direct
   non-GLM validation compilation were restored.
2. The authored-code-size gate counted generated build C files, so its result
   depended on which tests ran first. Build output is now excluded.
3. The old CUDA-performance source test referred to deleted implementation
   shapes and could not validate the current shared CUDA foundation. It now
   checks the current scale, TMA, packing, precision, cache, and topology seams.
4. The PTX capability test returned success when `ptxas` was absent and was
   therefore counted as a pass. The aggregate gate now records it as a CUDA-only
   skip.
5. Running aggregate build gates concurrently with `make clean` can produce a
   false failure by deleting a binary between build and execution. Final
   retained validation is serialized.
6. DeepSeek V4 Pro has a generated contract and CUDA compile surface but still
   has no complete independent executor.
7. Public headers remain substantially GLM-contaminated even though the neutral
   core archive and its actual include closure are clean.
8. CUDA 13 remains unavailable in this sandbox; host validation must not be
   presented as Blackwell qualification.
9. The repository's top-level `cache/` directory contains production source, not
   disposable build cache. Packaging and cleanup logic must preserve it and only
   exclude named generated-cache directories such as `__pycache__/`.
10. The GLM pipe simulator still used the removed `dspark_speculator` request-API
    field and therefore made a clean `make all` fail even though `make test` was
    green. It now uses the opaque `model_speculator` interface; clean `make all`
    is part of retained Phase 4 validation.

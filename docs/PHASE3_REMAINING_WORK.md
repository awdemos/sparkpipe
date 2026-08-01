# Remaining Work After Phase 3

## P0: shared CUDA execution foundation

1. Repair the shared quantized GEMM scale ABI. Activation UE4M3 bytes must not be consumed as FP32 scales, and row/expert/output/K-group scale indexing must be exact.
2. Repair the TMA protocol: CTA-wide producer ownership, correct mbarrier arrival semantics, and device-valid tensor-map lifetime.
3. Compile all CUDA with CUDA 13 for exact `sm_121a` and retain `ptxas` reports.
4. Add independent numerical vectors for BF16, FP8 and MXFP4 paths.

## P0: model execution closure

1. GLM 5.2: collapse the two diverged implementations into one link-complete production module; repair binder buffer preservation, per-layer context binding, route metadata, bulk prefill and FP8-expert/BF16-rest enforcement.
2. K3: connect the resident stage to the shipping driver, wire the AttnRes sideband, complete exact replay, and qualify the deliberate MXFP4-expert/BF16-activation variant.
3. Qwen 3.6 27B: prove BF16 end to end and connect restored work control to the production stage.
4. DSV4 Flash: wire the exact cache arena into the eventual execution module and implement every attention class, HCA/CSA compressor state, hyper-connections, hash-MoE and MTP.
5. DSV4 Pro: implement a separate Pro module with its generated geometry; do not reuse Flash layer-kind assumptions.

## P0: distributed queue correctness

1. Define one prepare/accept/execute/commit transaction across ranks.
2. Add request and step generations to every work packet.
3. Add downstream acknowledgements and reconnect deduplication.
4. Separate transport credit, resident reservation, execution occupancy and completion ownership.
5. Remove duplicate credit returns and lossy completion behavior.
6. Remove blocking waits and head-of-line stalls from event loops.

## P1: grouped-MoE production integration

1. Feed complete layer batches into sealed expert queues from the production scheduler.
2. Convert each sealed firing into an expert-major GPU descriptor without host synchronization.
3. Retain one expert firing per layer batch; fail or select a larger package bucket when one expert exceeds 1024 rows.
4. Measure route skew and replace the current 2x-mean tile heuristic with retained histograms.
5. Implement a weight-stationary large-queue variant for expert loads exceeding one M tile.
6. Overlap shared-expert work, route packing, expert weight streaming and route finalization.

## P1: DSV4 cache integration

1. Make the future DSV4 stage initializer call `SparkDsv4CacheArenaAllocate`.
2. Bind every layer to a `SparkDsv4LayerCacheArenaView` and reject class/ratio mismatches.
3. Add per-stage plans so PP placement allocates only the layers resident on that Spark.
4. Add cache-page lifecycle, reference counting, eviction and prefix-restore tests.
5. Measure whether CSA index history should remain 8-bit or use another checkpoint-native representation.

## P1: network

1. Bring up the direct single-rail ring first, with stage order matching physical adjacency.
2. Bring up one MikroTik 804 switch with one 100 Gbit/s rail per Spark.
3. Pre-advertise RDMA slots and remove the per-packet receive-ready round trip.
4. Add slot generations, bounded MR eviction, batched CQ polling and autonomous progress.
5. Rotate independent B1 boundary slots across available QPs without reordering one request.
6. Add device-direct buffers and validate GPUDirect visibility on the actual Spark/NIC/IOMMU stack.
7. Keep dual-switch dual-rail mode disabled until single-rail correctness receipts exist.

## P1: topology scheduler

1. Retain performance receipts by model, precision, context bucket, B bucket, PP degree, microbatch count and transport mode.
2. Measure ring PP placement before the switch arrives.
3. Measure one-switch PP8, PP13 and PP16 crossovers from B1 through B1024.
4. Treat a large pending queue as a sequence of qualified B buckets; do not merge all pending requests into one unbounded CUDA batch.

## Repository repair still required

1. `tests/test_glm52_ring_runtime.c` remains malformed and still calls an obsolete `SparkRingRuntimeBuildRankPlan` signature. The complete `make test` target now reaches this file after the stale stage-plan, TP-shard, hidden-transport and stagepack tests were repaired.
2. The GLM stagepack API validates the legacy NVFP4 contract but has no corresponding retained FP8-expert pack contract. The malformed orphan FP8 test block was removed; a real FP8 contract validator and test must replace it.
3. The production GLM module Makefile still references CUDA translation units absent from the attached source package, so exact link closure cannot be claimed.
4. DSV4 currently has only a validation doorway, not an executable resident-stage module. The exact allocator is ready, but no shipping stage exists to consume it yet.
5. Sealed GLM expert queues guarantee one firing per active expert only while each expert queue is at most 1024 rows. Larger qualified buckets require a different retained firing format or weight-stationary kernel.
6. The analytical active-expert model assumes independent uniform routing. Frozen routing bias and real traffic can produce materially different expert skew.
7. The repository's top-level `clean` target still depends on the absent `modules/glm52_w8lut_quality_weights` directory; clean validation currently removes `build/` directly.

## Repairs completed while advancing the full host suite

1. `tests/test_glm52_stage_plan.c` now uses the generic geometry-aware stage-plan ABI and GLM model constants.
2. `tests/test_glm52_tp_shard.c` now uses the generic TP-shard ABI and class constants.
3. `tests/test_hidden_transport.c` no longer references a deleted TCP transport module identifier.
4. The syntactically invalid orphan block in `tests/test_glm52_stagepack.c` was removed. It had no function declaration, no valid file-path arguments and no call from `main`.
5. The prior staging omissions of K3 generated contracts and Qwen work-control source are restored.
6. Generated DSV4 contract outputs were regenerated from their authoritative Flash and Pro contracts.

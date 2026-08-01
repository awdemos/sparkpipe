# Handoff — Phase 6 Transactional Completion

## Validate the source tree

```sh
make clean
make -j2 all
make -j2 test
sh tools/gates.sh
```

The exact CUDA qualification target is:

```text
CUDA 13.x
compute_121a PTX
sm_121a object/SASS and device link
```

Use `tools/cuda13_sm121a_compile_gate.sh` on a GPU-less Linux CUDA 13 host. A real Spark is still required for numerical execution, Compute Sanitizer, CUDA Graphs, transport ordering, and performance receipts.

## Mandatory model recipes

```text
Kimi K3
    routed experts: MXFP4 weights
    expert inputs:  BF16
    non-experts:    BF16
    accumulation:   FP32

GLM 5.2
    routed experts: FP8 E4M3 weights
    expert inputs:  BF16
    non-experts:    BF16
    accumulation:   FP32

Qwen 3.6 27B
    weights:        BF16
    activations:    BF16

DeepSeek V4 Flash / Pro
    routed experts: checkpoint-native FP4
    non-experts:    checkpoint-native FP8
    geometry:       separate generated Flash and Pro contracts
```

K3's upstream report describes MXFP4 expert weights with MXFP8 expert activations. SparkPipe's BF16-activation recipe is an intentional local deployment variant and requires its own numerical qualification.

## Phase 4 corrections

### Shared CUDA foundation

- Scale planes carry explicit encoding, capacity, group, row, and K-group geometry.
- UE4M3 activation scales are no longer reinterpreted as `float *`.
- Tensor maps are passed to kernels by value rather than through temporary host-stack pointers.
- TMA uses one CTA producer and an arrival count matching that producer contract.
- INT6/INT7 sign extension and scale application are explicit.
- Sub-byte quantization assigns each thread an exclusive eight-code byte range; overlapping read-modify-write stores are gone.

### Kimi K3

- Verification stores pre-convolution Q/K/V inputs and the exact transformed FP32 retention/write-gate values.
- Accepted-prefix folding consumes those exact values and does not repeat the approximate exponentials.
- The fold launches the recurrent kernel with the required FP32 dynamic shared-memory state tile.
- Fold arguments and replay capacities fail closed.
- The host slice equivalence test proves verify-plus-fold reaches the same state and convolution windows as a committed run.

### GLM 5.2

- Non-expert attention and dense/shared paths expose BF16-native GEMM entry points.
- Routed expert execution exposes FP8-weight/BF16-activation entry points.
- Model-description metadata forbids runtime precision substitution and hidden fallback.
- Required module identity names the actual FP8-expert/BF16-rest recipe.

### Qwen 3.6 and DSV4

- Qwen's BF16 contract is checked across configuration and binding surfaces.
- DSV4 Flash and Pro retain separate generated geometry headers.
- DSV4 Pro has an explicit CUDA compile surface, but not yet a shipping execution entry point.

## Next phase priorities

1. Define one prepare/accept/execute/commit protocol across all ranks.
2. Add request and step generations to every work packet and completion.
3. Add downstream acknowledgements and reconnect deduplication before local destructive state changes.
4. Separate transport-window credit, resident reservations, execution occupancy, and completion ownership.
5. Remove blocking waits from rank event loops and make completion draining backlog-aware.
6. Bring up the physical single-rail ring with PP order matching adjacency.
7. Then bring up one MikroTik 804 single-switch 100 Gbit/s fabric.
8. Pre-advertise RDMA slots, add slot generations, bounded MR eviction, batched CQ polling, and autonomous progress.
9. Compile and device-link every CUDA translation unit for exact `sm_121a`.
10. Run independent numerical references and sanitizers on a Spark.

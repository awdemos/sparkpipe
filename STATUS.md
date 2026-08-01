# SparkPipe Status — Phase 6 Transactional Completion

The authoritative deployment targets are:

- Kimi K3: MXFP4 routed-expert weights, BF16 expert activations, BF16 non-expert tensors, FP32 accumulation.
- GLM 5.2: FP8 E4M3 routed-expert weights, BF16 expert activations, BF16 non-expert tensors, FP32 accumulation.
- Qwen 3.6 27B: BF16 weights and activations, FP32 accumulation where required.
- DeepSeek V4 Flash and Pro: checkpoint-native FP4 expert and FP8 non-expert formats, each with a separate generated geometry contract.

The complete clean host build and test inventory passes. The architecture gate reports 69 pass, 4 CUDA-only skips, and 0 failures. The skips are not passes: CUDA 13, `compute_121a` PTX, `sm_121a` assembly/device linking, Blackwell numerical execution, race checks, and performance remain unmeasured.

Phase 4 repairs the shared scale ABI, TMA launch contract, sub-byte quantization write ownership, K3 exact speculative replay, GLM mixed-precision dispatch surfaces, Qwen BF16 source contract, and DSV4 Pro compile surface. It also removes generated build products from the authored-code-size metric.

This tree is a host-validated source candidate, not a production-qualified GPU package:

```text
HOST_BUILD_VALIDATED=true
HOST_TEST_INVENTORY_VALIDATED=true
ARCHITECTURE_GATES=69_PASS_4_CUDA_SKIP_0_FAIL
CUDA13_SM121A_COMPILE_NOT_RUN=true
BLACKWELL_EXECUTION_NOT_MEASURED=true
PRODUCTION_READY=false
```

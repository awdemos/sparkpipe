# CUTLASS header subset provenance

This directory contains the CUTLASS header subset required by SparkPipe's
FlashInfer SM120/SM121 grouped FP8 GEMM path.

Only the public headers needed by `flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`
are vendored:

```text
include/
tools/util/include/
```

The full upstream CUTLASS repository is not required at runtime. SparkPipe keeps
this subset in-tree so offline source archives can build the production FP8 MoE
CUDA path without fetching a nested submodule.

Upstream: https://github.com/NVIDIA/cutlass
License: see `LICENSE.txt`

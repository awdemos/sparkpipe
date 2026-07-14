# glm52_w8lut_quality_weights — candidate quality-max weight format (W8LUT v2)

Per-tensor exponent-biased 1-3-4 minifloat for routed experts: 8 b/w like production
FP8, measured 1.95x lower Frobenius error on real GLM-5.2 expert weights (2.64B params),
no block scales, no sidecars, one-add decode, bit-deterministic.

NOT a published module: reference semantics, offline measurement evidence, conversion
core, and gate harnesses for qualification. Plan and pack ABI delta:
docs/GLM52_W8LUT_QUALITY_FORMAT_20260714.md.

  make test_ref     G0: golden + property tests            [PASS, authoring machine]
  make crosscheck   G0b: numpy encoder == C reference      [PASS, authoring machine]
  make test_gpu     G1/G2 on a Spark                       [pending: CUDA not compiled by author]

source/w8lut.h,w8lut_ref.c      normative format reference
source/w8lut_kernels.cu         candidate expand + fused deterministic gemv
tools/glm52_w8lut_codec.py      vectorized encoder shared with the production packer
source/kl_eval.py               G4 logit-KL harness
source/realdata_result.json     retained offline measurement

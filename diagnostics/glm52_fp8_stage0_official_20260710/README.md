# GLM-5.2 FP8 stage-0 official comparison

This directory compares the SparkPipe six-layer stage `0:6` with the official
Transformers GLM-5.2 FP8 implementation for token IDs:

```text
45494 10397 13 10397 13
```

The official reference used:

```text
model: zai-org/GLM-5.2-FP8
image: sparkpipe/glm52-fp8-reference:transformers-5.12.1-kernels0.13-fgfp8v2
attention: eager
dtype: bfloat16
```

The SparkPipe reference used tiled MLA attention, eager stage execution, the
resident FP8 MoE packs, shared experts, and explicit accurate BF16 MoE
activations:

```text
GLM52_EXACT_PP13_ATTENTION_EXECUTION_MODE=1
GLM52_EXACT_PP13_DISABLE_GRAPH_REPLAY=1
GLM52_ENABLE_CUDA_GRAPH_REPLAY=0
SPARKPIPE_FP8_MOE_ACCURATE_BF16_ACTIVATION=1
```

`official/after_layer_N.bf16` contains five contiguous hidden rows from the
official layer hook. `sparkpipe/token_T_after_layer_N.bf16` contains one hidden
row from SparkPipe. All hidden rows contain 6144 BF16 values.

Reproduce the numeric comparison with:

```text
python3 tools/glm52_hash_diff.py --layer-numeric \
  diagnostics/glm52_fp8_stage0_official_20260710/official \
  diagnostics/glm52_fp8_stage0_official_20260710/sparkpipe
```

Observed error bands across the five rows:

| Layer | Relative L2 | Minimum cosine |
| --- | --- | --- |
| 0 | 0.009740 to 0.014493 | 0.999895 |
| 1 | 0.014718 to 0.016878 | 0.999861 |
| 2 | 0.014457 to 0.017255 | 0.999852 |
| 3 | 0.013210 to 0.017863 | 0.999842 |
| 4 | 0.015930 to 0.024160 | 0.999709 |
| 5 | 0.016118 to 0.025983 | 0.999662 |

The layer-3 route receipts use signed 32-bit expert IDs and float32 weights.
Tokens 1 through 4 select the same eight experts as the official model with
weights agreeing to approximately `1e-4`. Token 0 agrees on seven experts and
changes only the eighth near-tie.

The investigation established these transitions:

```text
missing shared expert:
  layer-3 relative L2 approximately 0.105 to 0.113

shared expert plus FP8 activation re-quantization:
  layer-3 relative L2 0.023549 to 0.072467

shared expert plus BF16 activations and FP8 weights:
  layer-3 relative L2 0.013210 to 0.017863
```

The exact DSA context-prefix bypass changed no output bytes for prompt tokens
0 through 3. At token 4 it produced relative L2 `0.0285` and cosine `0.999593`
after all six layers. DSA ordering still needs a long-context parity gate, but
it was not the short-prompt accuracy failure isolated here.

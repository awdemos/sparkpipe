# GLM52 active-row linear-plan correctness

Date: 2026-07-14

Hardware:

```text
host: spark0
gpu: NVIDIA GB10
driver: 580.159.03
cuda target: sm_121a
```

## Failure

The FP8 PP13 resident created deterministic BF16 cublasLt plans using the
maximum execution-row capacity. For the B64 plus MTP configuration this was
448 rows. Submissions with one active row then executed an algorithm selected
for 448 rows. The layouts could be changed to one row at launch time, but that
did not restore the serialized result because the selected algorithm was still
capacity-shaped.

## Fix

Each resident BF16 cublasLt plan is prepared for the exact execution-row count
before submission. A launch fails validation if its prepared row count does
not equal the submitted row count. There is no fallback path. The builder has
one pending work item, so a row-shape change cannot replace a plan while an
earlier submission is in flight.

The CUDA validator now accepts separate active count, row capacity, and PP13
bucket values. It can require byte identity with an expected hidden output.

## Results

All commands used FP8 stage payloads, tiled online-softmax attention, the
25-token serialized input, layers 0 through 5, and the B64 stage launcher.

```text
active=1 max=448 eager
submissions=25
total_us=380602.778
maximum_us=28917.113
output_sha256=b5177beae75f023f934bb33172d9d23110dab351c3e1913b2439a843febb61ab
oracle_sha256=b5177beae75f023f934bb33172d9d23110dab351c3e1913b2439a843febb61ab
expected_output_match=1

active=1 max=448 graph
submissions=25
total_us=376314.527
maximum_us=29562.936
graph_captures=1
graph_replays=25
output_sha256=b5177beae75f023f934bb33172d9d23110dab351c3e1913b2439a843febb61ab
expected_output_match=1

active=4 max=4 eager
submissions=25
total_us=485893.143
maximum_us=33486.188
output_sha256=433a8bf5d8505708fcb33b159cd1634b0e8beded8ed88254f5d1712a14391a68

active=4 max=448 eager
submissions=25
total_us=490890.387
maximum_us=33796.984
output_sha256=433a8bf5d8505708fcb33b159cd1634b0e8beded8ed88254f5d1712a14391a68
expected_output_match=1
```

The active-1 max-448 run also produced 150 layer boundaries and 750 phase
boundaries. `diff -qr` against the max-1 serialized oracle returned zero for
both trees. Those 900 raw files are included here. `SHA256SUMS` covers all 905
raw artifacts plus this README.

These timings include eager per-token stage execution and are correctness-gate
measurements, not serving throughput claims.

## Standard gate

```sh
make -C modules/glm52_resident_decode_stage validate \
    NVCC=/usr/local/cuda/bin/nvcc \
    CUDA_ARCH=sm_121a \
    MAX_STAGE_MICROSECONDS=1000000 \
    GLM52_REQUIRE_B12X_RESIDENT_PACK=0 \
    GLM52_VALIDATION_MODE=exact_pp13_stage_sequence \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    GLM52_VALIDATION_MAX_ACTIVE_SEQUENCE_COUNT=448 \
    GLM52_VALIDATION_EXACT_PP13_BATCH_BUCKET=64 \
    GLM52_EXACT_PP13_MODEL_QUANTIZATION=fp8 \
    GLM52_STAGE_PACK_DIR=/home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    GLM52_FP8_MOE_PACK_DIR=/home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    GLM52_MODEL_DIR=/home/spark0/models/hf/zai-org/GLM-5.2-FP8 \
    GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=0 \
    GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=6 \
    GLM52_PIPELINE_INPUT_HIDDEN_BF16=/path/to/stage_input_embedding_sequence.bf16 \
    GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=/tmp/stage0_active_rows.bf16 \
    GLM52_VALIDATION_EXPECTED_OUTPUT_HIDDEN_BF16=/path/to/oracle_after_layer_5.bf16 \
    GLM52_REQUIRED_CUDA_LINK_ARGS="<b12x-adapter.a> <b12x-backend.a> <b12x-table.a>"
```

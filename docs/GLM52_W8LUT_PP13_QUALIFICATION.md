# GLM-5.2 W8LUT PP13 Qualification

This procedure qualifies the GLM-5.2 configuration with BF16 non-expert
weights and W8LUT routed experts. It is deliberately split so artifact checks
can run while the production ring owns the GPUs. CUDA and service tests wait
for an explicit idle window.

## Coverage

The gates prove different things and are not interchangeable:

| Gate | What it proves |
| --- | --- |
| One-layer artifact preflight | StagePack schema, BF16 tensor contracts, source identity, W8 header and regions, E0 ranges, and sampled codes from all 768 W1/W2 expert routes |
| One-layer CUDA | Production attention, DSA, router, shared expert, W8 resident binding, W8 expert execution, and hidden output for one routed layer |
| Six-layer CUDA | Exact rank-local StagePack loading, layer chaining, and graph replay for one PP13 stage |
| Full-ring smoke | Rank placement, PP transport, all 78 layers, final head, and MTP |
| Evaluation | Model accuracy |

The first two gates cover most W8-specific code. They do not prove PP13
transport, cross-rank ordering, final-stage behavior, or end-to-end accuracy.

## Read-Only Artifact Gate

The preflight never opens CUDA, starts a service, stops a process, or writes to
the model directories. It validates one real layer on rank 2 with:

```sh
python3 tools/glm52_w8lut_artifact_preflight.py \
    --rank 2 \
    --layer 12 \
    --stagepack-root /home/spark2/sparkpipe_artifacts/glm52_w8lut_bf16_pp13_stage_payload_v1 \
    --w8lut-pack-root /home/spark2/sparkpipe_artifacts/glm52_w8lut_resident_moe_pp13_stage_v1
```

Omit `--layer 12` to require and validate all six W8LUT packs assigned to rank
2. Add `--verify-sha256` when the extra full-file read is wanted.

After the PR is merged and every rank has pulled the same clean commit, run the
parallel zero-drift check. It writes one JSON line as each rank completes:

```sh
python3 tools/glm52_w8lut_ring_preflight.py \
    --expected-commit <full-merged-commit> \
    --one-layer-per-rank \
    --receipt /private/tmp/glm52_w8lut_one_layer_ring.jsonl
```

Remove `--one-layer-per-rank` for the complete rank-local artifact gate. The
controller fails if a checkout differs from the exact commit, has tracked
changes, lacks its assigned artifact, mixes formats, or has a source identity
mismatch.

## Deferred One-Layer CUDA Gate

Run this only when spark2 has exclusive GPU ownership. The input must be a real
nonzero BF16 hidden vector with exactly 6,144 elements. The rank-2 StagePack
contains the BF16 trunk for layers 12 through 17; this gate selects layer 12
and one W8LUT expert pack.

```sh
make -j glm52_resident_decode_stage_firmware_package \
    GLM52_MOE_BACKEND=w8lut \
    GLM52_EXACT_PP13_MODEL_QUANTIZATION=w8lut \
    GLM52_MODEL_DIR=/home/spark2/sparkpipe_artifacts/glm52_w8lut_bf16_pp13_stage_payload_v1 \
    GLM52_STAGE_PACK_DIR=/home/spark2/sparkpipe_artifacts/glm52_w8lut_bf16_pp13_stage_payload_v1 \
    GLM52_W8LUT_MOE_PACK_DIR=/home/spark2/sparkpipe_artifacts/glm52_w8lut_resident_moe_pp13_stage_v1 \
    GLM52_VALIDATION_MODE=routed_from_hidden \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=12 \
    GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=1 \
    GLM52_PIPELINE_INPUT_HIDDEN_BF16=<real-layer-12-input.bf16> \
    GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=/private/tmp/glm52_w8lut_layer12_output.bf16 \
    GLM52_ENABLE_CUDA_GRAPH_REPLAY=1 \
    MAX_STAGE_MICROSECONDS=1000000
```

This command builds and validates the production module. It must report the
W8LUT model quantization and production W8 backend; a fallback backend is a
failure.

## Deferred Six-Layer Gate

Once all six rank-local W8 packs are present, change the CUDA gate to:

```sh
GLM52_VALIDATION_MODE=exact_pp13_stage_slice
GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=12
GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=6
```

Keep the same explicit W8LUT, StagePack, input, output, and graph settings. A
passing receipt must show `model_quantization=w8lut`, `layer_count=6`, nonzero
layer-body launches, and graph capture/replay counts.

## Release Assembly

W8LUT releases require both artifact roots. The assembler rejects a W8 release
if either is omitted and updates the gateway and resident roles together:

```sh
python3 tools/sparkpipe_release_assemble.py \
    --template <known-good-release> \
    --output <new-w8lut-release> \
    --release-id <release-id> \
    --git-commit <full-merged-commit> \
    --kv-logical-blocks <qualified-count> \
    --model-quantization w8lut \
    --stagepack-root '/home/{host}/sparkpipe_artifacts/glm52_w8lut_bf16_pp13_stage_payload_v1' \
    --moe-pack-root '/home/{host}/sparkpipe_artifacts/glm52_w8lut_resident_moe_pp13_stage_v1' \
    --mtp
```

Validate the immutable release with `sparkpipe_release_manager validate`, then
use the normal merged-main deployment procedure. Only the full-ring smoke and
evaluation can qualify it for production use.

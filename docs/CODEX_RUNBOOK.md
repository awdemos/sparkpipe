# Codex Runbook

Follow this workflow for SparkPipe changes. Do not substitute a Mac CUDA build,
a dirty Spark checkout, a copied shared object, or a compile-only result.

## Goal

The goal is correct, fast GLM-5.2 inference through the public API and all 13
Sparks. A build is not an inference test. A ready health endpoint is not an
inference test. A release is tested only after a real prompt returns correct
tokens through the installed release.

## Fixed Rules

- `main` is the advisor handoff branch.
- Publish repository changes with `updaterepo`; do not manually push.
- Run host tests with `make -j`.
- Build CUDA and releases on spark0 from a clean worktree at merged `main`.
- Keep rank-local FP8 packs at the one configured stable root.
- Never delete or regenerate `.sp*` packs without explicit user approval.
- Never silently fall back to a reference, compatibility, or demo path.
- Deploy one immutable manifest generation to every rank.
- Keep B16 while correctness or deployment is changing; increase the bucket
  only through a new manifest and a measured inference gate.

## Repository Update

Work in the checkout whose contents are the desired change. Run local tests,
then let the repository tool create, validate, merge, and synchronize the PR:

```sh
make -j test
updaterepo "Short human title"
```

Do not pass paths, validation commands, PATs, branches, or manual GitHub
commands to `updaterepo`. If it fails, fix the failing workflow step rather
than bypassing it.

## Spark0 Build Root

Use spark0 for CUDA and release work. The persistent checkout may be dirty, so
build from a detached clean worktree at the exact merged commit:

```sh
git -C /home/spark0/src/sparkpipe-main-live fetch origin main
git -C /home/spark0/src/sparkpipe-main-live worktree add --detach \
    /tmp/sparkpipe-release-<sha8> origin/main
git -C /tmp/sparkpipe-release-<sha8> status --short
git -C /tmp/sparkpipe-release-<sha8> rev-parse HEAD
```

The status output must be empty and the commit must equal merged `main`.

## Build Gates

Run the host suite, release tools, CUDA resident archive, backend, and
transport from the clean worktree:

```sh
make -C /tmp/sparkpipe-release-<sha8> -j \
    test tools \
    glm52_pp13_service_backend \
    hidden_transport_tcp_cuda \
    glm52_pp13_node_context_builder
```

Package the FP8 driver with the exact six-layer PP13 validator, not the legacy
dense-to-layer3 default:

```sh
make -C /tmp/sparkpipe-release-<sha8> -j \
    glm52_resident_decode_stage_firmware_package \
    MAX_STAGE_MICROSECONDS=1000000 \
    GLM52_VALIDATION_MODE=exact_pp13_stage_slice \
    GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
    GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=0 \
    GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=6 \
    GLM52_PIPELINE_INPUT_HIDDEN_BF16=<nonzero-12288-byte-fixture> \
    GLM52_ENABLE_CUDA_GRAPH_REPLAY=1 \
    GLM52_MODEL_DIR=/home/spark0/models/hf/zai-org/GLM-5.2-FP8 \
    GLM52_STAGE_PACK_DIR=/home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    GLM52_FP8_MOE_PACK_DIR=/home/spark0/models/sparkpipe/glm52_fp8_pp13_stage_payload_v1 \
    GLM52_EXACT_PP13_MODEL_QUANTIZATION=fp8 \
    GLM52_MOE_BACKEND=fp8 \
    GLM52_REQUIRE_B12X_RESIDENT_PACK=0
```

Require both validation lines: one for the archive and one for the linked
`model_driver.so`. Record stage time, graph captures/replays, nonzero output
count, and checksum.

## Assemble A Release

Clone the previous known-good role manifest and replace every rebuilt artifact
with the repo-owned assembler. It refuses unknown paths, recomputes every size
and SHA-256, writes into a temporary directory, and renames atomically:

```sh
python3 tools/sparkpipe_release_assemble.py \
    --template <known-good-release> \
    --output <new-release> \
    --release-id <release-id> \
    --git-commit <full-merged-sha> \
    --max-active 16 \
    --kv-pool-tokens 65536 \
    --replace bin/sparkpipe_release_manager=build/sparkpipe_release_manager \
    --replace bin/sparkpipe_glm52_cuda_residentd=build/sparkpipe_glm52_cuda_residentd \
    --replace bin/sparkpipe_glm52_pp13_rank_daemon=build/sparkpipe_glm52_pp13_rank_daemon \
    --replace bin/sparkpipe_glm52_http_gateway=build/sparkpipe_glm52_http_gateway \
    --replace lib/libglm52_pp13_node_context_builder.so=build/libglm52_pp13_node_context_builder.so \
    --replace lib/libglm52_pp13_service_backend.so=build/libglm52_pp13_service_backend.so \
    --replace lib/libhidden_transport_tcp_cuda.so=build/libhidden_transport_tcp_cuda.so \
    --replace lib/model_driver.so=build/packages/glm52_resident_decode_stage/stages/stage_000/model_driver.so
```

Validate before serving:

```sh
<new-release>/bin/sparkpipe_release_manager validate \
    --manifest <new-release>/sparkpipe.json
```

The 64K pool is the B1 correctness configuration when rank12 also owns native
MTP weights. Increase it only after measuring resident memory headroom; it is
not a B16 concurrency or long-context claim.

## Deploy The Ring

Serve the immutable release from spark0. Apply roles in this order:

1. `pp13_cuda_residentd` on ranks 0 through 12 concurrently.
2. Wait for every PID, Unix socket, rank-local pack, and CUDA allocation.
3. `pp13_rank_daemon` on ranks 1 through 12 concurrently.
4. `spark0_gateway` on rank 0.

Each role uses the installed release manager:

```sh
/home/<host>/sparkpipe_runtime/bin/sparkpipe_release_manager agent \
    --release-url http://spark0:<release-port>/ \
    --staging-dir /home/<host>/sparkpipe_state/release_staging \
    --install-dir /home/<host>/sparkpipe_runtime \
    --state-dir /home/<host>/sparkpipe_state \
    --host <host> \
    --rank <rank> \
    --role <role> \
    --once
```

Do not launch a second resident beside an existing allocation. The resident
agent must stop the old generation before starting the new one. Compare the
installed builder and driver hashes on all 13 ranks with the manifest.

## Actual Inference Gate

First require clean health:

```sh
curl -fsS http://spark0:18080/health
```

Then send a streaming, greedy request. Read the API key from the installed
file without printing it:

```sh
curl -sS -N --max-time 120 \
    -H "Authorization: Bearer $(cat /home/spark0/sparkpipe_runtime/API_KEY)" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","prompt":"Say OK. OK.","max_tokens":1,"temperature":0,"stream":true}' \
    http://127.0.0.1:18080/v1/completions
```

The correctness receipt is token `10397`, text `" OK"`, followed by a done
event. Also run a distinct factual prompt and an 8-or-more-token decode. A
queued `202` without token and done events is not a passed inference test.

## Performance Gate

Use `tools/sparkpipe_api_stress.py` for API timing and retain its JSONL and
summary. Report separately:

- exact six-layer CUDA stage time
- prefill time
- decode-step time
- first-token latency
- steady generated-token rate
- concurrent aggregate token rate
- transport time

Run concurrency 1 first, then 4, 16, and larger only while throughput rises.
If completion times form a staircase, requests are serialized before the GPU
scheduler and larger bucket claims are invalid.

## Diagnostics

Role logs are under:

```text
/home/<host>/sparkpipe_state/run/<role>.pid.log
```

When accuracy is under investigation, keep the detailed stage dump and hash
instrumentation enabled and compare the ring against the serialized FP8 oracle
at every stage. First divergence wins; do not guess downstream causes.

## Forbidden Evidence

- Mac CUDA output
- a dirty Spark checkout
- an unmerged branch or copied shared object
- `make -j test` without the SM121 archive/package gate
- health without a returned token
- a reference fallback presented as production performance
- a B64/B256/B1024 claim when the live request reaches one lane

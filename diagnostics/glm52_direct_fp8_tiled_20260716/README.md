# GLM-5.2 Direct FP8 Tiled KV Gate

This directory records the exact 13-stage CUDA gate for tiled online-softmax
attention backed by full FP8 E4M3 MLA, key-nope, and value caches. The active
rows are quantized directly into mapped cache slots; no full BF16 cache
duplicates are allocated.

## Source

```text
base commit: ba5ee0e
hardware: spark0 GB10, sm_121a
model: zai-org/GLM-5.2-FP8
stage shape: 13 stages x 6 layers
active sequences: 1
input token ids: 45494 10397 13 10397 13
attention: tiled online softmax
KV cache: full FP8 E4M3 with per-block FP32 scales
```

The source was compiled with nvcc for `sm_121a`. Each six-layer stage was
executed eagerly in order, with the previous stage output used as the next
stage input. The final stage was required to emit token `10397`; the gate
failed closed on any other token.

## Result

```text
13/13 stages passed
78/78 layers executed
final token: 10397
stage-0 eager: 89.324 ms for five submissions
stage-0 graph: 84.150 ms for five submissions
stage-0 graph captures: 1
stage-0 graph replays: 5
graph output vs eager output: byte-identical
```

`timings.tsv` contains every stage receipt. The stage-6 cold outlier includes
fixture setup and is not a steady-state performance claim.

## Numeric Receipt

`candidate_boundaries/` contains the 13 combined five-row BF16 boundaries.
`numeric.tsv` compares every row to
`diagnostics/glm52_b1_fp8_isolation_20260712/serialized`.

The direct FP8 cache is not byte-equivalent to the historical BF16-cache
oracle. Stage-0 cosine is at least `0.999244`; error accumulates for
history-dependent rows across 78 layers, while the exact expected final token
still passes. This is an exact-token and execution-path gate, not a
corpus-level accuracy claim.

The graph boundary is retained separately in `graph_boundary/`.

## Reproduce

```sh
python3 tools/glm52_hash_diff.py --layer-numeric \
    diagnostics/glm52_b1_fp8_isolation_20260712/serialized \
    diagnostics/glm52_direct_fp8_tiled_20260716/candidate_boundaries
```

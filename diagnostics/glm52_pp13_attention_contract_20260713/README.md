# GLM-5.2 PP13 Attention Contract Receipt

Date: 2026-07-13

## Finding

The accurate serialized FP8 stage validator and the distributed PP13 builder
were executing different attention implementations. The serialized B64
validator uses tiled online-softmax attention. The live builder selected
absorbed-latent attention.

This was isolated with the same FP8 stage payloads, the same B64 capacity, and
the same 25-token C-tokenized prompt input.

## Measured Evidence

Serialized B64 stage-0 output SHA-256:

```text
tiled/default: 2ac025ebe33bdd48fc4368557f7c6f501986c10516ca7ea4a0d914d7a18c9c10
absorbed:      b1db2b030a42b5121f863739204f78cf31389cadaa572740c0f94e1d5d6c0007
```

Re-running serialized stage 0 with absorbed attention made every one of the
25 stage-0 token outputs bit-identical to the current distributed rank-0
outputs: `max_abs=0`, `rel_l2=0`, and `cos=1` for all 25 rows. The raw result
is in `absorbed_vs_ring_stage0.tsv`.

Comparing the distributed absorbed run against the accurate tiled serialized
pipeline shows the error accumulating through the stage chain. The full raw
table is in `tiled_vs_ring.tsv`.

Transport was independently cleared in the same run: all 300 adjacent-hop
hidden comparisons matched.

## Fix Contract

The PP13 builder must:

- allocate expanded K-nope and V caches from the physical KV token capacity;
- wire both caches into every layer node context;
- select tiled online-softmax attention;
- require tiled attention during resident validation instead of accepting an
  absorbed or null-cache configuration.

At a 16,384-token physical pool, the added cache allocation is exactly:

```text
16384 tokens * 64 heads * (192 K-nope + 256 V) * 2 bytes
= 896 MiB per layer
= 5.25 GiB per six-layer rank
```

## Pre-Merge Build Receipt

The patched source passed:

```text
make -j test
make -C modules/glm52_resident_decode_stage -j archive \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
```

The exact six-layer FP8 package validator passed for both the static archive
and linked `model_driver.so`. Both produced:

```text
nonzero=6142
checksum64=6664805546341863234
layer_bodies=6
launch_chains=1
graph_captures=1
graph_replays=2
```

This document records the localization and build receipt. Correctness of the
distributed fix still requires a merged-main immutable release, real greedy
inference, and a new all-stage numeric comparison.

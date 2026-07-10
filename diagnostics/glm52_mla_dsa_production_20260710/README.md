# GLM-5.2 absorbed MLA and tiled DSA receipts

This directory records the pre-deployment CUDA receipts for the PP13 FP8
absorbed-MLA and DSA score/top-k implementation.

Hardware and toolchain:

```text
GPU: NVIDIA GB10
driver: 580.159.03
CUDA: 13.0.88
target: sm_121a
base main: aced2b2646bbdea1cd513f1bc726f02865868ec4
```

The matched MLA input contains five BF16 hidden rows for token IDs:

```text
45494 10397 13 10397 13
```

The exact six-layer stage `0:6` ran eagerly with FP8 attention and MoE
payloads, absorbed-latent attention, shared experts, and accurate BF16 MoE
activations. It completed five submissions in `368356.617 us`; the slowest
submission was `85017.906 us`.

Compare the raw layer outputs with the committed official FP8 oracle:

```text
python3 tools/glm52_hash_diff.py --layer-numeric \
  diagnostics/glm52_fp8_stage0_official_20260710/official \
  diagnostics/glm52_mla_dsa_production_20260710/sparkpipe_layers
```

All 30 token/layer boundaries are present. The worst final-layer relative L2
is `0.038615`; the minimum cosine is `0.999260`. The raw per-layer outputs are
under `sparkpipe_layers/`, and the attention, residual, normalization, MoE,
and routing receipts are under `sparkpipe_phases/`.

The DSA parity target exercises the production score kernel followed by the
exact radix top-k selector. Its synthetic score construction makes the exact
selected set known for every row. The measured receipts are in
`dsa_parity.txt`.

The production score workspace is one shared tile:

```text
16 rows * 1048576 candidates * sizeof(float) = 67108864 bytes
```

Candidate capacity is selected from the live context length: 2048 minimum,
then powers of two through 1048576. The score workspace is reused across row
tiles and all six layers. At B1024 this replaces six 4 GiB per-layer score
buffers with one 64 MiB stage workspace.

The memory-contract test rejects raw sideband widths, packet-capacity slack,
duplicate DSA tile constants, stale generated model geometry, and duplicate
prefill-only DSA score storage.

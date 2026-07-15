# GLM-5.2 B256 Compressed MLA Receipt

Date: 2026-07-15

The PP13 FP8 production builder was changed from full expanded BF16 K/V
storage to absorbed MLA with one 576-element BF16 cache row per physical
token. The pre-publish checks used current `main` at
`c803a03413e28feb07a7af707184232fa9756ab2` plus the scoped builder change.

## Numeric Gate

The exact stage-sequence validator ran stage `0:6` on GB10 for the five real
embedding rows corresponding to token IDs:

```text
45494 10397 13 10397 13
```

It used FP8 attention and MoE packs, eager execution, and absorbed MLA. All 30
token/layer outputs were compared with the committed official GLM-5.2 FP8
oracle using:

```text
python3 tools/glm52_hash_diff.py --layer-numeric \
  diagnostics/glm52_fp8_stage0_official_20260710/official \
  <absorbed-layer-dump-directory>
```

Results:

```text
missing boundaries:          0
worst final-layer rel_l2:    0.026662
minimum final-layer cosine:  0.999645
five-submit total:           105139.533 us
slowest submit:              33117.754 us
```

The linked `model_driver.so` repeated the gate successfully.

## B256 Resident Gate

Both preflight residents used:

```text
max_active_sequence_count = 256
execution_row_capacity = 1792
kv_pool_token_capacity = 1376256
MTP = enabled
```

Rank 0 reached `state=ready` with all 3 FP8 MoE layers and all 54 FP8 scaled
GEMM plans bound:

```text
builder allocation bytes: 18011532915
largest allocation bytes: 1903165440
CUDA bytes consumed:      19847274496
CUDA bytes free at ready: 3262906368
```

Rank 12 is the worst case because it owns six routed base layers, the MTP
layer, and the vocabulary heads. It reached `state=ready` with all 7 FP8 MoE
layers and all 58 FP8 scaled GEMM plans bound:

```text
builder allocation bytes: 20967580359
largest allocation bytes: 1903165440
CUDA bytes consumed:      94246797312
CUDA bytes free at ready: 774983680
full-vocab logits bytes:  634388480
MTP draft-head weights:   951582720
```

The largest builder allocation is the compressed MLA cache for one layer.
The old tiled builder attempted a 33,822,867,456-byte expanded key cache per
layer and could not initialize B256. Both preflight residents were terminated
after reaching ready, and host memory returned without swap growth.

These are initialization and numeric receipts, not end-to-end throughput
claims. Ring inference and DS4 evaluation require a merged-main release.

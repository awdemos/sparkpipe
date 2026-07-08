# GLM 5.2 PP13 Serialized vs Ring Differential - 2026-07-08

This bundle captures the current FP8 PP13 correctness gap with raw data for advisor review.

## Matched Prompt

Text: `Say OK. OK.`

Token IDs used by the serialized spark2 run:

```text
45494 10397 13 10397 13
```

The 13-rank HTTP gateway accepted the same text with `prompt_tokens=5` and returned token id `0`, text `!`.

## Key Result

The matched ring run proves rank2 receives prompt hidden values that still match serialized input embeddings, not the serialized output expected after ranks 0 and 1.

See:

```text
ring_pp13_matched_say_ok_ok/rank2_vs_serial_expected.tsv
```

For tokens 1-4:

```text
ring_rank2_rx == serialized_embedding: true
ring_rank2_rx == serialized_after_layer_11: false
ring_rank2_tx == serialized_after_layer_17: false
```

So the ring prefill path is passing hidden states through before/through rank2 instead of applying the stage-slice compute that the serialized spark2 path applies.

## Directory Contents

```text
serialized_spark2/
  tokens.txt
  *.bf16                         raw stage outputs for the five-token sequence
  sha256.txt                     file-level SHA256
  fnv64_by_token.tsv             per-token hashes using the hidden transport hash
  logs/                          raw serialized runner logs per stage

ring_pp13_matched_say_ok_ok/
  response.sse                   raw gateway SSE response for the matched prompt
  pre_counts.tsv                 log slicing anchors
  rank2_vs_serial_expected.tsv   direct matched differential table
  ring_logs/                     fresh request log slices; only rank2 had hash debug enabled

ring_pp13_full_trace_prior/
  rank0.log ... rank12.log       prior fully traced 13-rank chain
  chain_analysis.txt             pass-through analysis for that full-ring run
```

## Code Fix Included

The CUDA validation loader had an FP8 MoE header parser bug. It treated header field 3 as the scale block size, but production FP8 packs use field 3 as `maximum_active_sequence_count`. The validation parser now accepts packs when field 3 is at least the validation active sequence count.

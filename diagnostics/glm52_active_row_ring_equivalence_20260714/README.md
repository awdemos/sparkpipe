# GLM-5.2 FP8 Active-Row Ring Equivalence

This directory records the first byte-exact equivalence result between the
serialized FP8 reference and the 13-rank PP13 ring after the active-row linear
plan fix.

## Release

```text
commit: 7f4f1b5df2b133da2b36433955de2e7dc166f6b4
release: glm52-fp8-main-7f4f1b5d-b64-active-row
generation: 20260713214544
max active sequences: 64
KV pool tokens per rank: 16384
gateway logical KV blocks: 256
attention: tiled online softmax
model quantization: FP8
```

The immutable release manifest is `release_manifest.json`. Installed hashes
from all 13 ranks are in `installed_hashes.txt`.

## Exact Input

`chat25_request.json` was sent to `/v1/chat/completions` with greedy sampling.
The C tokenizer produced the 25 IDs in `chat25.tokens`. The corresponding
serialized embedding input is `stage_input_embedding_sequence.bf16`.

## Results

The observed receipts are:

```text
inter-rank hidden transfers: 300/300 byte-identical
rank outputs through layer 71: 300/300 byte-identical to serialized
rank12 layer-77 row hashes: 25/25 identical to serialized
chat25 serialized token: 785
chat25 ring token: 785 ("The")
Say OK. OK. ring token: 10397 (" OK")
8-token decode: accepted, 8 token events, done
post-run live requests: 0
post-run queued requests: 0
post-run event backlog: 0
post-run dropped events: 0
```

`numeric.tsv` contains all 300 numeric comparisons. Every row reports
`max_abs=0`, `rel_l2=0`, and `cos=1`; the final line is `jumps=0`.

This is measured equivalence for the retained prompts and boundaries. It is
not a corpus-level model accuracy score and it is not a performance result.
Phase hashing and raw hidden dumping were enabled for this run.

## Raw Data

`serialized_boundaries/` contains the 13 serialized six-layer outputs:

```text
after_layer_5.bf16
after_layer_11.bf16
...
after_layer_77.bf16
```

Each file contains 25 consecutive BF16 hidden rows of 12,288 bytes each.

`ring_boundaries/` contains 600 raw 12,288-byte buffers:

```text
rank0:       25 transmitted stage outputs
ranks 1-11: 25 received inputs plus 25 transmitted outputs per rank
rank12:     25 received inputs
```

The sequence ID is part of every filename. `stage12_ring_hashes.txt` and
`stage12_serial_hashes.txt` cover the final stage output because rank12 returns
tokens rather than transmitting its hidden output to another rank.

## Reproduce

From the repository root:

```sh
python3 tools/glm52_hash_diff.py --numeric \
    diagnostics/glm52_active_row_ring_equivalence_20260714/serialized_boundaries \
    diagnostics/glm52_active_row_ring_equivalence_20260714/ring_boundaries
```

The raw transport invariant is, for each token and rank 0 through 11:

```text
rankN_tx == rankN+1_rx
```

The serialized invariant is, for each token and rank 0 through 11:

```text
rankN_tx == serialized after_layer_(6*N+5) row
```

`raw_hop_integrity.txt` and `serial_boundary_identity.txt` record the counts
from those byte comparisons. `SHA256SUMS` covers every retained artifact.

## API Receipts

- `chat25.sse`: matched 25-token serialized prompt, token 785, done.
- `sayok.sse`: independent known-token gate, token 10397, done.
- `decode8.sse`: eight sequential decode tokens, done.
- `health_after.json`: clean drained gateway state after all three requests.

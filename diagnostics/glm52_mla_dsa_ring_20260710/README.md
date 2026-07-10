# GLM-5.2 PP13 absorbed MLA and DSA ring receipts

This directory records the merged-main 13-rank inference receipts for commit
`032eb9a114ef3598e1d70083c09ae06b35c3577c` and release
`glm52-fp8-main-032eb9a-b1-absorbed-mla-tiled-dsa`.

The release was rebuilt from merged main, validated, installed through the
release manager on all 13 Sparks, and started in dependency order: residents,
rank daemons, then the Spark0 gateway. Every resident reported `state=ready`.
Rank 0 used 66,895 MiB of VRAM, routed ranks used approximately 93,272 MiB,
and the final rank used 95,087 MiB.

## Inference

The OpenAI-compatible streaming endpoint was called with greedy sampling:

```text
prompt: Say OK. OK.
prompt token ids: 45494 10397 13 10397 13
max_tokens: 1
temperature: 0
result token id: 10397
result text: " OK"
```

Two consecutive runs produced the same token and byte-identical hidden output
at all 78 recorded rank/token boundaries. `run1/` contains sequence 2 and
`run2/` contains sequence 3. `determinism.txt` records all 78 matching SHA-256
digests. `api_ok.sse` is a separate successful streaming request captured from
the same release.

A second prompt produced a distinct expected token:

```text
prompt: Say YES. YES.
result token id: 14071
result text: " YES"
```

An 11-token prompt followed by eight decode steps also completed. The final
token ids were `11012 25 154842 2132 5868 1075 697 1943`. The gateway trace is
in `multitoken_trace.txt`.

The final health response in `health.json` reports an empty queue and backlog,
zero dropped events, both readiness bits set, and no blocker.

## Ring integrity

`trace/` contains the transport hash lines from the current resident process on
every rank. Each rank emitted 49 frames. The 12 transport hops therefore cover
588 send/receive pairs. `hop_integrity.txt` reports no missing or mismatched
pair. `chain.txt` reports `findings=0`: no zero hidden payload and no stage
passthrough was observed.

Re-run the chain check from the repository root with:

```text
python3 tools/glm52_hash_diff.py --chain \
  0:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank0.log \
  1:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank1.log \
  2:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank2.log \
  3:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank3.log \
  4:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank4.log \
  5:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank5.log \
  6:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank6.log \
  7:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank7.log \
  8:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank8.log \
  9:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank9.log \
  10:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank10.log \
  11:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank11.log \
  12:diagnostics/glm52_mla_dsa_ring_20260710/trace/rank12.log
```

## MLA accuracy

The rank-0 output after its exact six-layer stage was compared with the
committed official FP8 oracle for all five prompt tokens. `stage0_numeric.txt`
reports no missing boundaries. The worst relative L2 is `0.038615`, and the
minimum cosine similarity is `0.999260`. These values exactly reproduce the
pre-deployment serialized CUDA receipt in
`diagnostics/glm52_mla_dsa_production_20260710/mla_layer_numeric.txt`.

`mla_probe.txt` records the live MLA cache and RoPE-pair hashes after each
request. The repeated greedy requests produce stable values.

## DSA correctness

The live ring exercised the DSA selected-token transport on every applicable
share-group boundary. `dsa_sideband.txt` contains 72 send/deliver records for
the matched request. Every record uses the symbolic selected-token payload
size:

```text
2048 selected indices * sizeof(uint32_t) = 8192 bytes
```

The sideband hashes are nonzero, token-dependent, deterministic between the
two matched runs, and identical at each send/receive hop.

The exact CUDA score plus radix-top-k parity receipts remain in
`diagnostics/glm52_mla_dsa_production_20260710/dsa_parity.txt`. They cover one
and 16 active rows at 1,048,576 candidates, plus 17 rows at 4,096 candidates to
cross the 16-row score tile. The production implementation therefore has both
a live-ring integration receipt and an exact 1M-candidate selector receipt.

## Raw data

`run1/` and `run2/` contain the BF16 hidden payload at every PP13 transport
boundary for the five prompt tokens and first decode token. Ranks 0 through 11
use their transmitted stage output; rank 12 records the received final-stage
input. `sha256.txt` covers every binary:

```text
shasum -a 256 -c diagnostics/glm52_mla_dsa_ring_20260710/sha256.txt
```

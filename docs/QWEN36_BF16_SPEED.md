# Qwen 3.6 27B, full BF16 on the sparkring - the speed model
Grounded in inference/llms/qwen_3_6/config.h. Dense hybrid: 64 layers,
16x (3 gated DeltaNet -> 1 gated full attention), hidden 5120, FFN
17408, GQA 24/4 heads x 256, GDN 16 key x 48 value heads x 128.

Parameters, counted from the projections: full-attention layer 372.2M,
DeltaNet layer 383.1M, embedding 1.271B (x2 untied), one MTP layer ->
**27.27B total, 54.5 GB in BF16**. Fits one GB10 with room; the whole
ring is a luxury, not a requirement.

Per-token streamed state on top of weights, per sequence:
- GDN: 0.541 MB/layer read+write x 48 layers = **51.9 MB/seq/token**
  (context-independent - the linear layers never grow).
- Full-attention KV: 4 KB per context token per layer x 16 layers =
  **ctx x 64 KB/seq read per token** (BF16 KV, the growth term).

Decode step time = (54.5 GB + B x (52 MB + ctx x 64 KB)) / bandwidth.
At 273 GB/s per GB10:

| nodes (BW)      | ctx   | B=1  | B=8   | B=32   | B=64   |
|-----------------|-------|------|-------|--------|--------|
| 1  (0.27 TB/s)  | 1k    | 5.0  | 39    | 150    | 281    |
|                 | 4k    | 5.0  | 38    | 135    | 233    |
|                 | 16k   | 4.9  | 34    |  97    | 138    |
| 4  (1.09 TB/s)  | 4k    | 20   | 153   | 539    | 931    |
| 13 (3.55 TB/s)  | 1k    | 65   | 512   | 1946   | 3654   |
|                 | 4k    | 65   | 497   | 1753   | 3027   |
|                 | 16k   | 64   | 447   | 1254   | 1794   |

Readings:
- **Single-stream floor: 5 tok/s on one node, 65 tok/s across the
  ring** (200 ms vs 15 ms per token of pure weight streaming). BF16 is
  a bandwidth tax paid in latency; interactive single-user wants the
  ring or a quantized ladder rung.
- **The batch knee is where B x state rivals weights**: at 4k context
  each lane adds ~314 MB/step, so past B~170 on the ring the state
  stream overtakes the weights and per-lane throughput halves - the
  table's B64 column is still weights-dominated everywhere.
- Long context bites only the 16 full-attention layers: 16k context
  costs 1 GB/seq/step - the 3:1 hybrid doing exactly its job, and
  16k/B64 on the ring still clears 1.7k tok/s.
- These are bus-saturation ceilings assuming the cohort-13 pipeline
  keeps the bus busy (audit F1-F3 are the risks to that assumption);
  compute rides under the weight stream at these batch sizes.

Debug-distance status: qwen has firmware config, host geometry,
doorway, null provider link, conformance rows, and - with the
uniform-estimated profile - an ADMITTED scheduler path proven by
test_uniform_profile_admit on qwen's exact dense geometry {64,64}.
Getting here surfaced and fixed two real balancer defects: the cut
rule that pinned the dense prefix to stage zero forbade every split in
a fully dense model, and the reachability probe indexed the maximum
layer count instead of the geometry's - an out-of-bounds read past the
VLA that glm had been passing on stack luck. Both are gated now. What
remains for qwen tokens is the same execute rung K3 waits on, plus the
chat surface.

# GLM5.2 → K3: every difference

The glm52 resident decode stage was the template for this module: same ABI, same
frame/admission/snapshot surface, same stage-pack idea, same launcher split
between a C host module and one `.cu`. This file is the list of everything that
had to change and why, plus the guesses and where each one is fixed when the K3
report lands on 2026-07-27.

Provenance tags: **D** = disclosed by Moonshot, **G** = our guess.

---

## 1. Attention: DSA sparse selection → 3:1 KDA / gated NoPE MLA  (D)

glm52 is DeepSeek-V3.2-lineage: every layer is full MLA, with a DSA indexer
picking 2048 tokens per query. K3 is Kimi-Linear-lineage: three Kimi Delta
Attention layers per one full-attention layer.

This is the deepest change in the module, because it changes what *state* means:

| | glm52 | k3 |
|---|---|---|
| per-sequence state | KV cache only | KDA recurrent state **and** MLA latent cache |
| state size | grows with tokens | KDA is **fixed** (dk·dv·heads·layers), MLA grows |
| addressable by prefix | yes, blocks are content-addressed | **no** for KDA: the state is a fold of the whole prefix |
| prefill | one pass, all tokens | KDA must be **chunked** (64) and carried in order |
| a rejected/resampled token | drop a block | must **replay**, the fold is not invertible |

Consequences that are now module structure, not detail:

- `SparkK3KdaStatePool`: fp32 state, `dk*dv` per head per layer per lane, plus
  `state_cold_by_row` so a fresh sequence starts from zero without a memset.
  At 54 KDA layers × 64 heads × 128×128 × 4 B = **226 MiB per lane**, resident,
  never paged. That is the price of linear attention and it is charged per lane
  whether or not the sequence is short.
- `SparkK3LaunchKdaChunk` for prefill, `SparkK3LaunchKdaDecodeStep` for the
  single-token path. Two kernels, one recurrence: the cpu reference checks they
  are the same model (`spark_k3_reference.c`, 3.1e-6).
- **No prefix caching for KDA.** glm52 can serve a shared system prompt from
  cached blocks. K3 cannot, for the 54 KDA layers: it must fold the prefix per
  sequence. The MLA latent cache still blocks and could still be shared; this
  module does not implement sharing (see STATUS.md).

**G**: the 3:1 period, and that MLA is the *last* layer of each period
(`SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE 3`). Fix: `spark_k3_model.h:57-58`, one
line each; `SPARK_K3_MODEL_LAYER_IS_KDA` derives from them and every count in
the module and both JSON files derives from that.

## 2. MLA: RoPE → NoPE, and per-head output gates  (D)

- `rope_dimension` 64 → **0**. Kimi Linear's argument is that the KDA layers
  carry position, so the full-attention layers do not need RoPE. The whole
  fused-RoPE/KV-write path from glm52 is **gone**: `SparkK3MlaKvWriteKernel`
  just writes the normalized latent.
- Per-head sigmoid gate on the attention output (**Gated MLA**, D). New tensor
  `MLA_HEAD_GATE` (64×7168), applied in `SparkK3MlaValueUpKernel`. glm52 has no
  equivalent.
- Head dims: qk_nope 192 → **128**, v 256 → **128**, query_a 2048 → **1536**
  (**G**, K2 lineage). Latent stays 512.
- `MLA_QK_SCALE` is 1/√128, not 1/√(192+64): with no rope split there is one
  scale and no partial-dimension bookkeeping.

**G** fix sites: `spark_k3_model.h:103-108`. If K3 keeps RoPE, this is not a
one-liner — the kv-write and absorb kernels need the rope path back, which is
the one guess in this module that is expensive to be wrong about.

## 3. MoE: 256/top-8 → 896/top-16, SwiGLU → SiTU  (D)

- Expert count 256 → **896**, top-k 8 → **16**, shared expert kept (**G**: 1).
  The router's argmax loop is 16 sequential passes over a 896-wide shared array
  (`SparkK3MoeRouteKernel`); glm52's was 8 over 256.
- **SiTU**: `sigmoid(gate) * tanh(up)` replaces SwiGLU's `gate*sigmoid(gate)*up`.
  One line in `SparkK3MoeExpertInterKernel`. It is not a cosmetic swap: SiTU is
  **bounded** (|out| ≤ 1) where SwiGLU is unbounded, which is the whole point
  of it for a QAT'd MXFP4 model — activations that cannot blow up quantize.
  The cpu reference pins both the bound and the fact that the two functions
  are far apart (max gap 34).
- `routed_scaling_factor` 2.5 and top-k normalization carry over from K2 (**G**).
- **Quantile Balancing** and **Per-Head Muon** are disclosed but are
  *train-time*. They are deliberately **not implemented** and must not be: a
  driver that "balanced" the router at inference would be a different model.
  Noted in the descriptor as `train_time_only_not_implemented`.

## 4. Quantization: NVFP4 (group 16) → MXFP4 (group 32) + MXFP8 activations  (D)

glm52's fast path is NVFP4 with an fp8 second-level scale and group 16. K3 is
MXFP4: E2M1 payload, **E8M0 per-32 scale**, no second level. The decode helpers
(`SparkK3DecodeE2m1`, `SparkK3DecodeE8m0`) are simpler than glm52's for exactly
that reason.

MXFP8 activations are disclosed but this module keeps activations in **bf16**.
That is a deliberate divergence: MXFP8 activation quantization is a throughput
optimization, and shipping it before there is a device to measure it on would
be optimizing a number nobody has seen. The tensor contract records the
disclosed format; the module does not pretend to implement it.

## 5. Residuals → Block Attention Residuals  (D) — the serving-level change

glm52: `x = x + attn(norm(x))`. K3 replaces the residual *read* with a softmax
mixture over a stack of representations:

```
candidates = completed_blocks[0..b) + running_partial
score_i    = pseudo_query · (key_norm ⊙ rmsnorm(candidate_i))
mixed      = Σ softmax(score)_i · candidate_i
```

Two sites per layer (before attention, before mlp), a learned pseudo-query per
site, and the block boundary appends the partial and starts a new one. The
embedding is block 0.

Structural consequences:

- `attnres_representations_bf16` is `MAX_REPRESENTATIONS × hidden` per row, not
  one hidden vector. The partial is not a separate buffer: it *is* the last live
  representation slot. (An earlier revision kept a separate partial buffer, and
  the mix kernel would have read an uninitialized candidate on the block-opening
  layer. The cpu geometry oracle now pins peak candidates ≤ budget.)
- **The pipeline-parallel transport payload changes shape.** glm52 hands the
  next stage one hidden vector. K3 must hand it the *whole stack*: completed
  blocks plus the partial — and the stack **grows** along the pipeline (1 block
  at layer 0, 9 at layer 71). At 7168 × bf16 that is 14 KiB → 143 KiB per token
  across a stage boundary. This is why this module is **single-node v1** and
  rejects hidden transport (`ValidateSliceIsWholeStack`, hard SCHEMA_ERROR, no
  silent fallback). A PP13 K3 needs a transport format that carries the stack;
  that is the first thing to build after the report lands.

**G**: block span 9 layers (`spark_k3_model.h:132`). Disclosed is "~8 blocks";
9 layers × 8 blocks = 72 = the guessed layer count, and the geometry oracle
confirms 8 blocks open and the final mix sees 10 candidates against a budget of
10. If the real span differs, `ATTNRES_BLOCK_LAYERS` is the one line, and
`MAX_REPRESENTATIONS` must be re-derived with it (the `#error` in the header
catches the inconsistent pair).

## 6. Everything else

| | glm52 | k3 | tag |
|---|---|---|---|
| hidden | 6144 | 7168 | G |
| layers | 78 | 72 (71 routed + 1 dense) | G |
| first routed layer | 3 | 1 | G |
| vocab | 154880 | 163840 | G |
| dense intermediate | 12288 | 18432 | G |
| speculative decode | dspark, 6 draft tokens | **none** | — |
| stage pack | per-variant dirs | one `.k3sp` file, 144 B header, 56 B entries | — |
| resident bytes | fits PP13 at nvfp4 | **1398 GiB** at mxfp4 | measured |

The 2.8T identity is the only reason to believe the hidden/layer/expert guesses
at all: `7168 × 2048 × 3 × 896 × 71 = 2.802T` total and `16 × 71 = 50.03B`
active. Change any one of those five numbers and it stops closing.

MTP: glm52 carries a whole speculator (dspark, markov rank, anchors). Nothing
about K3 speculation is disclosed, so this module has **no** draft path and
reports `draft_token_count = 0`. Inventing one would have been the easiest thing
here to fake and the least defensible.

## 7. Frame contract: batched resident (buffer_count == 0) → 2-buffer single sequence

glm52 rejects any frame carrying buffers: tokens live in resident device state
and the frame names lanes. K3 v1 takes the opposite contract — `buffer_count == 2`,
host token ids in, host sampled ids out, one sequence per frame, lane =
`driver_dispatch_slot`.

This is a **deliberate divergence, and it is the module's main serving-side
limitation**: it means no cross-sequence batching, so 896 experts get activated
for one token at a time and the MoE path is bandwidth-bound on a batch of 1.
It is the right v1 shape (a bring-up driver you can drive from a test harness
without a resident token store) and the wrong production shape. Batching is
tracked in STATUS.md, not silently absent: `ValidateFrameShape` rejects
`new_token_count > 1` on decode and `active_slot_count > 1` explicitly, with a
unique negative code, rather than mis-serving.

---

## Guess ledger — where each one is fixed

| Guess | Site | Cost if wrong |
|---|---|---|
| hidden 7168 / layers 72 / moe_inter 2048 / experts 896 | `spark_k3_model.h:30-44` | one line each; the `#error` pairs catch inconsistency; pack + docs re-derive |
| first routed layer 1 | `spark_k3_model.h:32` | one line |
| attention period 4, phase 3 | `spark_k3_model.h:57-58` | one line; all counts derive |
| KDA 64 heads, dk=dv=128, low-rank 128 | `spark_k3_model.h:68-71` | one line each; **dk=dv is `static_assert`ed** in the gate kernel's fused loop |
| **log decay = −softplus(low_rank(x))** | `spark_k3_resident_decode_stage_cuda.cu:284,332` | one line — **but see the clamp below** |
| MLA dims (query_a 1536, qk_nope 128, v 128) | `spark_k3_model.h:103-108` | one line each |
| **MLA rope = 0 (NoPE)** | `spark_k3_model.h:107` | **not a one-liner**: needs the rope path restored in kv-write and absorb |
| AttnRes block span 9 | `spark_k3_model.h:132` | one line + re-derive `MAX_REPRESENTATIONS` |
| vocab 163840, eos 163585 | `spark_k3_model.h:35,155` | one line; overridable at pack build |
| shared expert count 1 | `spark_k3_model.h:43` | one line |

### The one guess that is not just a number: the KDA decay regime

The chunk kernel floors the *running* log decay at `MIN_LOG_DECAY = -16` because
`kh` carries `exp(-running)` in bf16. The cpu reference proves this is exact
while the floor is not reached (**3.1e-6** against the sequential recurrence,
flat across carried chunks) — and proves it is **wrong** once the floor engages
(**3.77**, because the gram coupling `exp(running_c - running_r)` collapses
toward 1 and distant tokens couple as if nothing decayed between them).

At `-16` over a 64-token chunk, the plan needs **mean log decay > -0.25/token**.
Whether K3 sits in that regime is unknown until the report. If it does not, the
fix is `KDA_CHUNK_TOKENS` or `MIN_LOG_DECAY` — **not** the test tolerance. Both
regimes are pinned as permanent tests
(`SparkK3ReferenceCheckKda`, `SparkK3ReferenceCheckKdaDecaySaturation`) so this
cannot regress into a silent accuracy bug.

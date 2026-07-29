# sparkpipe

**Frontier open-weights models on prosumer metal.** A from-scratch C/CUDA
serving engine for a ring of NVIDIA DGX Spark (GB10) nodes, built around one
thesis: on unified-memory consumer silicon, *inference is a bytes problem*.
Every design decision here is downstream of counting bytes on the memory bus
and launches on the stream, and deleting both.

Primary target: **Kimi K3** — 93 layers of hybrid KDA linear attention + MLA,
MXFP4 experts, million-token context on constant per-sequence state — served
across **16× GB10** (2 TB unified memory, 273 GB/s per node, 200 GbE RDMA
ring, 29 µs measured hop floor). Also runs GLM 5.2 in production today, with
GLM 5.5, DeepSeek V4 GA, and Qwen 3.8 drivers tracking their releases.

This README describes the system we are releasing toward. Anything described
here and not yet true is enumerated in [`docs/techdebt.md`](docs/techdebt.md)
— the README is the contract, techdebt is the diff.

---

## Why this exists

A 1T-parameter MoE at MXFP4 is ~560 GB of weights. That does not fit in any
consumer GPU, barely fits in one H200 node you cannot buy, and fits with room
to spare in sixteen 128 GB unified-memory boxes that plug into wall power.
The catch: GB10 is **memory-bus-bound** (273 GB/s LPDDR5X against ~31 dense
TFLOPS bf16 and a tensor-core stack that wants FP4). Batch-1 decode arithmetic
intensity is ~1 FLOP/byte; the bus is the whole game.

So sparkpipe is organized as a bandwidth ledger with an engine attached:

- **Weights** are read once per step, in the narrowest format the accuracy
  budget allows (MXFP4/NVFP4 experts, FP8-E4M3 or W8LUT projections, bf16
  where it matters), through GEMM kernels whose epilogues do the graph's
  adds for free.
- **Activations** never take a round trip the dataflow doesn't require. The
  AttnRes partial-sum rides GEMM epilogues (`accumulate_bf16`: the module
  output projection folds into the partial in its own store — ~271 kernel
  launches and several full-hidden round trips deleted per K3 decode token).
- **KV** is paged in the model's own geometry (MLA compressed latents are
  576 × bf16/fp8 per token, not a key/value pair), and KDA recurrent state
  lives in a fixed-slot pool because a token-paged arena is the wrong
  allocator for linear attention.
- **Launches** are treated as bytes' evil twin: per-token tile-prefix
  rebuilds (700/token) gone, add kernels (271/token) gone, CUDA-graph
  capture of the remaining ~700 GEMM launches is the next line in the
  ledger.

## The machine

```
16× DGX Spark GB10
  ├─ 128 GB LPDDR5X unified, 273 GB/s
  ├─ Blackwell tensor cores (FP4/FP8/BF16), sm_121a
  └─ 200 GbE ConnectX RDMA, ring topology
Ring aggregate: 2 TB memory, 4.37 TB/s bus, 29 µs/hop software floor
Parallelism: PP16 (pipeline), TP×PP for wide layers; EP deliberately dropped
```

Expert parallelism was measured and rejected: at GB10's link budget,
shuffling activations to experts loses to shipping every expert's bytes
through the local bus exactly once. Pipeline parallelism with per-step cohort
re-formation keeps the hop cost at one 14 KB hidden-state transfer per stage.

## K3, first-class

K3 is the model this engine is shaped around, because its architecture is
the first frontier design that *agrees* with prosumer memory:

| Property | Value | Why it matters here |
|---|---|---|
| Layers | 93 = 69 KDA + 24 MLA (every 4th + last) | only 24 layers pay per-token KV reads |
| Hidden | 7168 | 14 KB/token inter-stage transfer |
| Experts | 384 routed, top-8 + shared, MXFP4 | ~0.53 B/param; all-expert sweep amortizes at batch |
| MLA cache | 512 latent + 64 unrotated (NoPE) | 27.6 KB/token/layer at bf16 — 44× smaller than full KV |
| KDA state | 96 heads × 128 × 128 f32 + conv windows | **434 MB/sequence, constant in context** |
| Context | 1M-class | state does not grow; only 24 MLA layers page |

The KDA/MLA split is priced by two allocators: the token-paged arena
(`cache/`, layout `MLA_COMPRESSED`) and the fixed-slot state pool
(`SparkStatePool`, O(1) acquire/release, no malloc, exhaustion is an
admission failure not an OOM). 434 MB/sequence is a real admission
constraint — ~280 sequences/node from state alone — and the scheduler
prices it.

### Performance model (estimates, bandwidth-derived)

Per-token active bytes at B=1: routed experts 8×3×7168×2048 × 0.53 B ×
85 MoE layers ≈ 15.9 GB, plus ~3.5 GB bf16 attention/shared/dense ≈
**19–20 GB/token**. Per node under PP16: ~1.2 GB → 4.4 ms; pipeline
traversal:

- **B=1: ~12–14 tok/s.** Marginal, honest, and interactive-usable with
  DSpark speculation riding on top.
- **B=256, 8K ctx: ~500 tok/s aggregate.** At 256 tokens/step all 384
  experts are touched per layer, so each node reads its resident ~6 layers'
  full expert set (~56 GB) + KDA state RW (~14 GB) + MLA KV (~3 GB) ≈ 73
  GB/step ≈ 267 ms ideal → ~960 tok/s at MBU 1.0; at the 0.5–0.6 MBU this
  kernel stack sustains, **≈480–580 tok/s**.
- KDA state traffic is linear in batch (no cross-sequence reuse) and
  context-constant; MLA KV reads grow with context — FP8 KV halves them,
  and past ~32K they become the wall. The ledger says so before the
  hardware does.

These are estimates from the bus model, marked as such until the 16-node
bring-up numbers replace them (`docs/BANDWIDTH_LEDGER.md` carries the
measured column).

## Architecture

```
sparkpipe/
├── api/gateway/          HTTP + SSE, request API, per-request lifecycle
├── scheduler/            cohort formation, prefix reuse, admission pricing
├── node/                 backend pump, rank daemon, ring transport
├── cache/                kv machinery: paged arena, tiering, prefetch lanes,
│                         JIT stage budgets, prefix cache, Mooncake KV tier
├── serving/              TP shard geometry, capacity estimation
├── inference/
│   ├── kernels/          Lm kernel family: GEMM (epilogue accumulate,
│   │                     per-thread E8M0 scale cache), norms, rope, sampling
│   ├── stage/            resident decode stage: module lifecycle, dispatch,
│   │                     runner — model-agnostic, linked per family
│   └── llms/             per-model layer/slice/engine (kimi_k3, glm5_2,
│                         deepseek_v4, qwen3_6 → 3_8, mimo25)
├── model-families/       serving-tier geometry per family (k3, glm52):
│                         kv geometry headers, drift-gated against kernels
├── modules/              per-family stage builds — THE LINKER SEAM
├── runtime/              pack IO, artifact check, CUDA resident IPC
├── speculation/          DSpark draft backend, MTP           (see techdebt)
├── quant/                MXFP4 · NVFP4 · FP8-E4M3 · W8LUT packers (see techdebt)
├── tools/                build.sh, gates.sh, packers, route collection
├── tests/                52 gates: host-CUDA recorders, drift gates, DRY law
└── docs/                 SERVING_PIPELINE · MODULE_MAP · BANDWIDTH_LEDGER · techdebt
```

### The serving pipeline

`gateway → scheduler → JIT weight residency → micro-batch dispatch →
resident stage → emit`, with three properties worth naming:

1. **Slot refill by construction.** Dispatch is a per-step re-formation,
   not a persistent cohort: a completed lane is simply absent from the next
   batch, a newcomer rides the prefill reserve (`QUEUE_DEPTH = cohort + 1`,
   statically asserted). Refill latency is the newcomer's prefill plus one
   pump — a floor, not a tax.
2. **Fast is the default.** Cross-sequence prefix reuse, async release,
   DSpark speculation: on unless a named kill-switch
   (`SPARKPIPE_DISABLE_*`) turns one off, and every mode announces itself
   in the startup banner. A disabled speed booster is a fallback that
   became a default; we don't ship those.
3. **Speculation rides existing traffic.** DSpark drafts travel on decode
   completion events — zero extra ring round-trips for speculative tokens.

### The linker seam

The stage module (`inference/stage/module.c`, 2,650 lines) is
model-agnostic and compiled **once per model family** by
`modules/resident_decode_stage_rules.mk`. The model's side of the contract
is two functions the linker resolves:

```c
SparkStatus SparkResidentDecodeStageModelValidateNodeContext(
    const SparkResidentDecodeStageNodeContext *);
SparkStatus SparkResidentDecodeStageModelValidateSliceNodeContext(
    const SparkResidentDecodeStageSliceNodeContext *,
    const SparkResidentDecodeStageNodeContext **);
```

GLM 5.2 answers with 1,577 lines of projection catalogs and MoE plan
checks; K3 answers in its own geometry. No ops tables, no registries, no
function pointers in wire ABI structs — the link step was always the seam.
The same pattern holds one layer down: KV machinery takes geometry through
a request struct (`latent + rope`, head dims, index-key schedule), and a
drift gate holds each family's serving-tier numbers equal to its kernel
config. Two tiers describing the same model, or refusing to build.

### Gates, or: how this stays true

52 gates run on every push, designed so that **CPU-only development can
prove CUDA dataflow**:

- **Host recorders**: the real layer/slice code runs on host with GEMMs
  recorded (synthetic outputs) and every other kernel executed for real.
  The K3 slice gate replays the byte-level partial/bank/aux/fold
  trajectory — it stayed green *through* the epilogue-fusion rewiring,
  which is the point of having it.
- **Drift gates**: serving-tier geometry vs kernel config, parsed from
  both headers, every dimension pair and slab formula.
- **The DRY naming law**: every glm reference in a common path is
  budgeted with a reason; budgets only shrink. Seam debt is a number in a
  test, not a feeling.
- **Coverage is itself gated**: when a refactor found that nothing
  compiled `cache/kv_cache.c` or `inference/stage/module.c`, the fix
  shipped with a gate so the hole stays closed.

## Model matrix

| Model | Status | What it exercises |
|---|---|---|
| **Kimi K3** | primary target | KDA linear attention + state pool, MLA latent paging, MXFP4 all-expert sweeps, NoPE |
| **GLM 5.2** | in production on the ring | full serving stack, DSA sparse index, B12x/FP8/W8LUT plan families, DSpark |
| GLM 5.5 | driver tracking release | successor to the production path |
| DeepSeek V4 GA | driver at structural parity | MLA + DSA index cache at GA scale |
| Qwen 3.8 | driver tracking release | dense+MoE hybrid, long-context GQA |

One serving stack, N model drivers. That sentence is the acceptance
criterion for the whole architecture (`docs/MODULE_MAP.md`), and every
model added is a test of it.

## Status

Production GLM 5.2 serving is live on the ring. K3 bring-up lands with the
16-node hardware window. The precise, honest gap between this document and
`git HEAD` lives in [`docs/techdebt.md`](docs/techdebt.md) — read it before
filing an issue that says the README lied.

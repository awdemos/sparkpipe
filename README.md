# sparkpipe

**Frontier open-weights models on prosumer metal.** A from-scratch C/CUDA
serving engine for a ring of NVIDIA DGX Spark (GB10) nodes, built around one
thesis: on unified-memory consumer silicon, *inference is a bytes problem*.
Every design decision is downstream of counting bytes on the memory bus and
launches on the stream, and deleting both.

Primary target: **Kimi K3** — 93 layers of hybrid KDA linear attention + MLA,
MXFP4 routed experts, million-token context on constant per-sequence state —
served across up to **16× GB10**. The mandatory support set also includes
GLM 5.2 with FP8 routed experts and BF16 elsewhere, Qwen 3.6 27B in BF16,
and separate DeepSeek V4 Flash and Pro drivers.

This README describes the release endpoint. The precise gap between it and
`git HEAD` lives in [`docs/techdebt.md`](docs/techdebt.md) — the README is
the contract, techdebt is the diff.

---

## Why this exists

A 1T-class MoE is ~1.6 TB of weights at sparkpipe's quantization ladder.
That fits in no consumer GPU and no single node you can buy — and fits with
its KV budget intact across sixteen 128 GB unified-memory boxes on wall
power. The catch: GB10 is **memory-bus-bound** (273 GB/s LPDDR5X against
~31 dense bf16 TFLOPS and a tensor-core stack built for FP4). Batch-1
decode arithmetic intensity is ~1 FLOP/byte; the bus is the whole game.

So sparkpipe is a bandwidth ledger with an engine attached:

- **Weights** are read once per step in the narrowest format the accuracy
  budget allows — MXFP4/NVFP4 experts, INT8/INT7/INT6 or FP8-E4M3
  projections, bf16 where it matters — through GEMM kernels whose epilogues
  do the graph's adds for free.
- **Activations** never take a round trip the dataflow doesn't require:
  the AttnRes partial rides GEMM epilogues (`accumulate_bf16` — module
  output projections fold into the partial in their own store; ~271
  launches and several full-hidden round trips absent per K3 decode token).
- **KV** pages in the model's own geometry (an MLA token is a 576-wide
  compressed latent, not a key/value pair). KDA recurrent state lives in a
  fixed-slot pool: a token-paged arena is the wrong allocator for linear
  attention.
- **Launches** are bytes' evil twin: tile-prefix rebuilds and add kernels
  deleted; CUDA-graph capture of the remaining GEMM launches closes the
  ledger.

## The machine

```
Up to 16× DGX Spark GB10
  ├─ 128 GB LPDDR5X unified, 273 GB/s
  ├─ Blackwell tensor cores (FP4/FP8/BF16), sm_121a
  └─ two 100 GbE ports per node
Aggregate at 16 nodes: 2 TB unified memory, 4.37 TB/s local memory bus
```

Bring-up is deliberately staged. The first debug topology is a single-rail
physical ring, with pipeline rank order following physical adjacency. The
first switched topology uses one MikroTik 804 and one 100 GbE port per Spark.
A second switch and the second port form a future independent rail only after
the single-rail transport has retained correctness and performance receipts.
The dual-rail mode is therefore represented in configuration but fails closed
rather than being selected implicitly.

## K3, first-class

K3 is the model this engine is shaped around, because its architecture
agrees with prosumer memory:

| Property | Value | Why it matters here |
|---|---|---|
| Layers | 93 = 69 KDA + 24 MLA (every 4th + last) | only 24 layers pay per-token KV reads |
| Hidden | 7168 | 14 KB/token inter-stage transfer |
| Experts | 896 routed, top-16 + 2 shared, MXFP4 routed weights | grouped expert queues amortize each active expert load across the layer batch |
| MLA cache | 512 latent + 64 unrotated (NoPE) | 27.6 KB/token/layer bf16, half at fp8 |
| KDA state | 96 heads × 128 × 128 + conv windows | constant in context; priced at admission |
| Context | 1M-class | state does not grow; only 24 layers page |

Memory budget at the 16-node endpoint: ~1.6 TB weights, ~400 GB for KV +
state. The state pool prices sequences at admission (f32 slabs are 434
MB/sequence; the bf16-state option halves that and doubles ceiling
batch), and the arena prices context: 2K-context batch ceilings sit in
the hundreds-to-~1.5K sequences, 256K context admits ~100.

## Topology is a scheduling decision

Weights may be placed with tensor, pipeline, or mixed parallelism, but no one
topology is declared globally optimal. SparkPipe qualifies a topology for an
exact tuple:

```text
model and precision recipe
context bucket
active-batch bucket
pipeline degree
microbatch count
transport mode
```

The debug ring prioritizes deterministic adjacency and observability. The
single-switch fabric then permits direct rank-to-rank links and a wider search
over PP and TP placement. Deep pipelines such as PP16 are credible for a
large queued workload because bubbles and per-hop latency can be amortized;
small interactive batches may prefer a different placement. Those crossovers
must come from retained B1-through-B1024 receipts, not from the old analytical
tables that assumed a smaller K3 expert pool and a different network.

Within one layer batch, routed MoE execution is weight-stationary: route all
tokens, group rows by expert, load each active expert once, run its grouped
GEMM, and scatter/fold the outputs back to token order. The scheduler does not
replay an expert-weight sweep for every 128-token chunk.

## Architecture

```
sparkpipe/
├── api/gateway/          HTTP + SSE, request API, per-request lifecycle
├── scheduler/            cohort formation, prefix reuse, admission pricing
│                         (KV blocks + state slabs), topology-aware dispatch
├── node/                 backend pump, rank daemon, topology-selected transport
├── cache/                kv machinery: paged arena, tiering, prefetch lanes,
│                         JIT stage budgets, prefix cache, Mooncake KV tier
├── serving/              TP shard geometry, capacity estimation
├── inference/
│   ├── kernels/          Lm kernel family: GEMM (epilogue accumulate,
│   │                     per-thread scale cache), norms, rope, sampling
│   ├── stage/            resident decode stage: module lifecycle, dispatch,
│   │                     runner — model-agnostic, linked per family
│   └── llms/             per-model layer/slice/engine (kimi_k3, glm5_2,
│                         deepseek_v4, deepseek_v4_pro, qwen_3_6, mimo_2_5)
├── model-families/       serving-tier geometry per family, drift-gated
│                         against the kernel configs
├── modules/              per-family stage builds — THE LINKER SEAM
├── runtime/              pack IO, artifact check, CUDA resident IPC
├── speculation/          DSpark draft backend, MTP
├── quant/                MXFP4 · NVFP4 · INT8/INT7/INT6 · FP8-E4M3 packers
├── tools/                build.sh, gates.sh, packers, route collection
├── tests/                67 source/host gates plus complete host-suite coverage
└── docs/                 SERVING_PIPELINE · MODULE_MAP · BANDWIDTH_LEDGER · techdebt
```

### The serving pipeline

`gateway → scheduler → JIT weight residency → micro-batch dispatch →
resident stage → emit`, with three properties worth naming:

1. **Slot refill by construction.** Dispatch is a per-step re-formation,
   not a persistent cohort: a completed lane is absent from the next
   batch; a newcomer rides the prefill reserve (`QUEUE_DEPTH = cohort +
   1`, statically asserted). Refill latency is the newcomer's prefill
   plus one pump — a floor, not a tax.
2. **Fast is the default.** Prefix reuse, async release, speculation: on
   unless a named kill-switch (`SPARKPIPE_DISABLE_*`) turns one off, and
   every mode announces itself in the startup banner. A disabled speed
   booster is a fallback that became a default; we don't ship those.
3. **Speculation rides existing traffic.** DSpark drafts travel on decode
   completion events — zero extra fabric round-trips for candidates.

### The linker seam

The stage module (`inference/stage/module.c`) is model-agnostic and
compiled **once per model family** by `modules/resident_decode_stage_rules.mk`.
The model's side of the contract is two functions the linker resolves:

```c
SparkStatus SparkResidentDecodeStageModelValidateNodeContext(
    const SparkResidentDecodeStageNodeContext *);
SparkStatus SparkResidentDecodeStageModelValidateSliceNodeContext(
    const SparkResidentDecodeStageSliceNodeContext *,
    const SparkResidentDecodeStageNodeContext **);
```

GLM answers with its projection catalogs and MoE plan checks; K3 answers
in its own geometry. No ops tables, no registries, no function pointers
in wire ABI structs — the link step is the seam. One layer down, the same
shape: KV machinery takes geometry through a request struct, and a drift
gate holds each family's serving-tier numbers equal to its kernel config.
Two tiers describing the same model, or refusing to build.

### Gates, or: how this stays true

67 host/source gates plus the complete host suite, designed so **CPU-only development proves CUDA
dataflow**: host recorders run the real layer/slice code with GEMMs
recorded and every other kernel executed, replaying byte-level
partial/bank/aux/fold trajectories; drift gates parse serving and kernel
headers into the same numbers; the DRY naming law budgets every glm
reference in a common path with a reason, and budgets only shrink; and
compile coverage is itself gated, so a hole found once stays closed.

## Model matrix

| Model | Role | What it exercises |
|---|---|---|
| **Kimi K3** | mandatory target | KDA state/replay, Gated MLA, Block AttnRes, top-16-of-896 Stable LatentMoE |
| **GLM 5.2** | mandatory target | BF16 non-expert path, FP8 routed experts, DSA, MTP and DSpark |
| **Qwen 3.6 27B** | mandatory target | BF16 full-attention/GDN execution and work control |
| **DeepSeek V4 Flash** | mandatory target | class-exact sparse/cache plan, FP4 experts and FP8 non-expert linears |
| **DeepSeek V4 Pro** | mandatory target | independent Pro geometry and execution surface |
| MiMo 2.5 | supported family | full/sliding attention and grouped MoE contracts |

One serving stack, N model drivers. That sentence is the acceptance
criterion for the architecture (`docs/MODULE_MAP.md`); every model added
is a test of it.

## Status

The gap between this document and `git HEAD` is
[`docs/techdebt.md`](docs/techdebt.md), in ledger order, with the exit
rule that nothing leaves it silently.

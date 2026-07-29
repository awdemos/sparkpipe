# sparkpipe

**Frontier open-weights models on prosumer metal.** A from-scratch C/CUDA
serving engine for a ring of NVIDIA DGX Spark (GB10) nodes, built around one
thesis: on unified-memory consumer silicon, *inference is a bytes problem*.
Every design decision is downstream of counting bytes on the memory bus and
launches on the stream, and deleting both.

Primary target: **Kimi K3** — 93 layers of hybrid KDA linear attention + MLA,
MXFP4 experts, million-token context on constant per-sequence state — served
across **16× GB10**. GLM 5.2 runs on the same stack, with GLM 5.5, DeepSeek
V4 GA, and Qwen 3.8 drivers tracking their releases.

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
16× DGX Spark GB10
  ├─ 128 GB LPDDR5X unified, 273 GB/s
  ├─ Blackwell tensor cores (FP4/FP8/BF16), sm_121a
  ├─ 200 GbE RDMA ring          — the bandwidth plane (25 GB/s/link, 29 µs hop)
  └─ 100 GbE switched fabric    — the latency plane (~20 µs all-reduce floor)
Aggregate: 2 TB unified memory, 4.37 TB/s bus
```

Two fabrics, two jobs. Bulk transfers — pipeline hidden-state hops, KV
migration, weight staging — take the ring. Tensor-parallel all-reduces take
the switch, whose worth is its latency floor, not its bandwidth. Expert
parallelism is deliberately absent: at these link budgets, shuffling
activations to experts loses to reading every resident expert through the
local bus exactly once.

## K3, first-class

K3 is the model this engine is shaped around, because its architecture
agrees with prosumer memory:

| Property | Value | Why it matters here |
|---|---|---|
| Layers | 93 = 69 KDA + 24 MLA (every 4th + last) | only 24 layers pay per-token KV reads |
| Hidden | 7168 | 14 KB/token inter-stage transfer |
| Experts | 384 routed, top-8 + shared, MXFP4 | all-expert sweep amortizes with microbatch |
| MLA cache | 512 latent + 64 unrotated (NoPE) | 27.6 KB/token/layer bf16, half at fp8 |
| KDA state | 96 heads × 128 × 128 + conv windows | constant in context; priced at admission |
| Context | 1M-class | state does not grow; only 24 layers page |

Memory budget at the 16-node endpoint: ~1.6 TB weights, ~400 GB for KV +
state. The state pool prices sequences at admission (f32 slabs are 434
MB/sequence; the bf16-state option halves that and doubles ceiling
batch), and the arena prices context: 2K-context batch ceilings sit in
the hundreds-to-~1.5K sequences, 256K context admits ~100.

## Topology is a scheduling decision

Weights place as **TP_g × PP_s** (g·s = 16): TP splits every tensor g
ways inside a stage, PP splits the 93 layers into s stages. Per-node
weight residency is identical in every topology; what moves is *where the
time goes*, governed by two laws:

1. **Expert amortization follows the microbatch: m = B/s.** A stage's
   weight pass serves only its microbatch, so touched experts per layer
   are 384·(1−(47/48)^m). Larger g ⇒ smaller s ⇒ larger m ⇒ fewer weight
   bytes per token. At m ≥ ~300 the sweep is total and weight cost per
   token collapses ~12×.
2. **All-reduce cost follows the group: 186·(20 µs + bytes(m,g)/12.5 GB/s)
   per layer-pass over the switch.** Nearly flat in g at decode message
   sizes; linear in m — which is exactly why prefill inverts the ranking.

Decode, aggregate tok/s (ideal / sustained at MBU 0.55, net eff 0.8;
2K ctx unless marked; 256K column at fp8 KV):

| topology | B=1 | B=8 | B=64 | B=1024 | B=64 @ 256K |
|---|---|---|---|---|---|
| TP16      | **50 / 30** | **111 / 63** | **202 / 115** | **1028 / 649** | **174 / 98** |
| TP8 × PP2 | 28 / 16 | 99 / 56 | 161 / 90 | 894 / 520 | 142 / 79 |
| TP4 × PP4 | 15 / 9 | 82 / 46 | 138 / 76 | 581 / 324 | 124 / 68 |
| TP2 × PP8 | 8 / 4 | 61 / 34 | 121 / 67 | 344 / 190 | 110 / 61 |
| PP16      | 4 / 2 | 32 / 18 | 106 / 58 | 223 / 123 | 97 / 54 |

Prefill, aggregate tok/s (2048-token chunks, compute-side):

| topology | prefill |
|---|---|
| PP16 | **11.8K** |
| TP2 × PP8 | 10.2K |
| TP4 × PP4 | 6.1K |
| TP8 × PP2 | 3.7K |
| TP16 | 2.1K |

Decode is monotone toward TP because the switch flattened the AR tax;
prefill is monotone toward PP because chunk-sized all-reduces are
bandwidth, and hidden-state hops are 14 KB. There is no single winner —
**the topology is chosen per deployment for its prefill:decode ratio**,
with TP8×PP2 and TP2×PP8 as the shoulders and the scheduler's chunked
prefill interleaving decode cohorts on whichever placement is loaded.
DSpark speculation is a *throughput* amplifier on this model — at full
expert sweep, verify rows ride weight reads already paid for (~1.9×
effective at ceiling batch); at B=1 on a top-8-of-384 MoE it is
approximately free but approximately nothing. All numbers are
bus-model-derived and labeled so until `docs/BANDWIDTH_LEDGER.md`'s
measured column replaces them.

## Architecture

```
sparkpipe/
├── api/gateway/          HTTP + SSE, request API, per-request lifecycle
├── scheduler/            cohort formation, prefix reuse, admission pricing
│                         (KV blocks + state slabs), topology-aware dispatch
├── node/                 backend pump, rank daemon, dual-fabric transport
├── cache/                kv machinery: paged arena, tiering, prefetch lanes,
│                         JIT stage budgets, prefix cache, Mooncake KV tier
├── serving/              TP shard geometry, capacity estimation
├── inference/
│   ├── kernels/          Lm kernel family: GEMM (epilogue accumulate,
│   │                     per-thread scale cache), norms, rope, sampling
│   ├── stage/            resident decode stage: module lifecycle, dispatch,
│   │                     runner — model-agnostic, linked per family
│   └── llms/             per-model layer/slice/engine (kimi_k3, glm5_2,
│                         deepseek_v4, qwen3_8, mimo25)
├── model-families/       serving-tier geometry per family, drift-gated
│                         against the kernel configs
├── modules/              per-family stage builds — THE LINKER SEAM
├── runtime/              pack IO, artifact check, CUDA resident IPC
├── speculation/          DSpark draft backend, MTP
├── quant/                MXFP4 · NVFP4 · INT8/INT7/INT6 · FP8-E4M3 packers
├── tools/                build.sh, gates.sh, packers, route collection
├── tests/                52 gates: host-CUDA recorders, drift gates, DRY law
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

52 gates on every push, designed so **CPU-only development proves CUDA
dataflow**: host recorders run the real layer/slice code with GEMMs
recorded and every other kernel executed, replaying byte-level
partial/bank/aux/fold trajectories; drift gates parse serving and kernel
headers into the same numbers; the DRY naming law budgets every glm
reference in a common path with a reason, and budgets only shrink; and
compile coverage is itself gated, so a hole found once stays closed.

## Model matrix

| Model | Role | What it exercises |
|---|---|---|
| **Kimi K3** | primary target | KDA state pool, MLA latent paging, MXFP4 all-expert sweeps, NoPE |
| **GLM 5.2** | proving stack | full serving path, DSA sparse index, quantized plan families, DSpark |
| GLM 5.5 | driver on release | expected to inherit the glm module wholesale with a new geometry header — that expectation is a test of the seam |
| DeepSeek V4 GA | driver at parity | MLA + DSA index cache at GA scale |
| Qwen 3.8 | driver on release | dense+MoE hybrid, mixed-layer cohort math |
| MiMo 2.5+ | driver at parity | SWA cache tier, compact-model fast path |

One serving stack, N model drivers. That sentence is the acceptance
criterion for the architecture (`docs/MODULE_MAP.md`); every model added
is a test of it.

## Status

The gap between this document and `git HEAD` is
[`docs/techdebt.md`](docs/techdebt.md), in ledger order, with the exit
rule that nothing leaves it silently.

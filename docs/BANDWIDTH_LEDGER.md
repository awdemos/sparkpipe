# Bandwidth ledger

GB10 law: the machine is memory-bus-bound, so the implementation's job is to
move the fewest bytes and keep the bus saturated with them - every byte doing
maximum work. This ledger itemises where bytes go per step, names the code
that spends more than the arithmetic requires, and ranks the fixes. Exact
tonnage comes from `tools/k3_param_budget.py`; formulas here are per token,
per layer unless said otherwise. R = read, W = write, all through the bus.

## K3 decode, batch B, per token

The irreducible stream (per token, whole model, TP1 view; divide weight terms
by tp_degree under TP):

  non-expert weights      every bf16 projection + norms, read once per step,
                          amortised across B - the dominant term at B=1
  expert weights          top_k unique-expert inflation x (w1|w3 + w2)
                          MXFP4 payloads + E8M0 planes, amortised across the
                          rows that share an expert
  lm head                 vocab x hidden bf16 per STEP - 2.35 GB at B=1 TP1,
                          147 MB per rank at TP16; the single largest
                          non-expert tensor and it amortises only across B
  KDA state               32 KB R + 32 KB W per sequence per KDA layer
                          (69 layers: 4.4 MB R+W per seq) - the recurrence's
                          floor, already minimal
  conv windows            3 x kernel x dim x 2B per seq per KDA layer - noise
  MLA latent cache        context x (kv_lora + rope) x 2B read per seq per
                          MLA layer + one row W - the long-context term
  activations             hidden/latent rows between kernels; small at decode

## The suspects (correct code that spends extra bytes or launches)

Ranked by estimated cost x confidence. "Fix window" says what this container
can do versus what needs the sparks.

S1  GATHER DOUBLE-TOUCH (route expansion). LmGatherRowsKernel copies each
    routed row: latent R + packed W, then the expert GEMM reads packed again.
    2R + 1W per routed byte where a gather-aware A-load would pay 1R.
    Cost: packed_rows x 3584 x 2B extra R+W per MoE layer - at B=16 decode
    ~1.8 MB/layer, ~150 MB/token across 82 MoE layers; at prefill C=512 it
    is ~117 MB/layer, ~10 GB per chunk (~20 MB/token) - the same order as
    the amortised weight stream. Fix: an indirect-A variant of the
    weight-only GEMM (cp.async per-row gather instead of TMA on A).
    Window: design now, measure on hardware - TMA vs ldgsts occupancy is
    the open question. UPDATE B-16s: at small B the gather is latency-hidden
    behind weight reads; the prefill case is the one that pays.

S2  GLM52 LAYER AUDIT DEBT. The glm path has not had the K3-grade byte
    audit. Known from reading: Glm52BindLayer clears buffers with memsets on
    every bind and rebuilds per-layer state K3 caches; the layer's gather
    and prefix behaviour predates this branch's GEMM fixes (it inherits the
    dense-derive automatically, but nothing told its route path about
    prefix_built). Window: full audit is CPU work - do it this week, it is
    the other first-class model.

S3  ATTNRES PARTIAL TOUCHES. K3PartialAdd is a separate kernel: hidden R +
    partial R + partial W per module add, 4 touches per layer. The adds
    could ride the producing GEMM's epilogue (accumulate-into-partial),
    saving one full hidden-row R+W per module - ~57 KB x rows x 2 per
    layer, ~10 MB/token at B=1 across 93 layers. Real but small; epilogue
    fusion is invasive. Window: hardware week, after profiles say whether
    the bus or the SMs notice.

S4  HOST-STAGED F32 ALL-REDUCE. SparkTpCollectiveAllReduceSumF32 is TCP
    from host memory in f32. Decode payloads are bf16 device tensors:
    today's path pays device->host staging plus 2x wire width. At decode
    the AR is latency-bound (fine); at TP prefill the 2x wire width is
    ~0.4 ms/token of pure format. Fix: bf16 sum variant + device staging
    (rdma.cu is the transport tier). Window: hardware.

S5  LAUNCH COUNT, REMAINDER. After 39b3b27 the prefix launches are gone;
    what remains is one launch per GEMM plus the fixed kernels - ~700 per
    K3 decode token. At ~3-5 us each that is 2-3 ms against an 85 ms B=1
    budget today, and the whole budget at TP16's ~5 ms target. Fix: CUDA
    graphs per (rows, layer-range) shape, captured at engine level.
    Window: hardware week; the engine's step planner was built to make
    step shapes repeat exactly so capture can work.

S6  VERIFY REPLAY STORES. Five LmCopyRowsKernel calls per KDA layer during
    verify - ~96 KB x rows per layer R+W. Verify-only, off the decode path,
    and the alternative (recompute at fold) re-reads weights instead.
    Verdict: correct spend, keep.

S7  HEAD EVERY STEP. The lm head reads its full vocab shard per step. At
    TP16 that is 147 MB/rank/step - second only to expert flow at small B.
    No cheat preserves exact sampling; DSpark amortises it across accepted
    tokens (one head read verifies eight positions), which is another
    reason speculation matters at B=1. Verdict: architectural, mitigated
    by DSpark.

## Prefill addendum

Chunked TP prefill's extra terms: the AR per layer (hidden x C x 2B x 2
transfers, S4's format tax on top), the gather at chunk width (S1's worst
case), and the MLA latent cache growing quadratically in reads across
chunks - the last is the DCP/position-sharding conversation, out of scope
for this ledger.

## Standing rule

A new kernel or path lands with its line in this ledger: what it reads,
what it writes, and why that is the minimum. The gates prove correctness;
this file is where "correct but slow" goes to be seen.

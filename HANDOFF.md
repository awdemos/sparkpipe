# Handoff: first-party kernels, PR #515

74 commits. The tree went from 425,650 code lines to 94,493 and gained a kernel
library that five models share. `third_party` is gone — 2,048 files, 1,235,873
lines — and so are the 27,307-line decode stage and the 10,628-line node context
builder that replaced it.

Read `tools/inventory.sh` output before anything else; it lists every file with
one line on what it does.

## Verify it still works

```
sh tools/get_cuda.sh      # nvcc, ~90 seconds, no driver, no root
sh tools/gates.sh         # 18 gates
sh tools/breakdown.sh     # code by module
sh tools/metric.sh        # code vs tests vs docs vs diagnostics
```

`tools/build.sh` compiles five unity builds plus the serving adapter for
`sm_121a`. **The arch flag is `-gencode arch=compute_121a,code=sm_121a`.**
`-arch=sm_121a` silently emits `.target sm_121` and drops every
architecture-specific instruction; the build asserts the emitted target rather
than trusting the flag.

## What the structure is

```
inference/kernels/   16 files, 3,704 lines. Every GPU kernel in the tree.
inference/llms/      5 models: config.h, unity.cu, layer.cuh, bind.cu
inference/stage/     the stage interface and serving adapter
api/                 request surface, gateway, serving engine, tokenizer
scheduler/           admission, batching, work control, speculation
cache/               cache.h is the new one; two legacy files remain
ring/                how nodes talk: transport, collectives, sidebands
node/                what runs on one: rank daemon, residentd
runtime/             pack loading, launch planning, workspace, json
```

The rule: device code in `inference/kernels/`, a weight format is a file in
`inference/kernels/formats/`, a model is a directory in `inference/llms/`, the
host is everything else by what it does.

## The five things most worth knowing

**Everything decodes to BF16.** GB10's BF16 ridge is 573 FLOP/byte and decode
arithmetic intensity is 8 to 64, so compute is free by about sixty times. A
format's job is to be narrow in memory and hand back a BF16 register. That is
why there is one GEMM where the old tree had seven.

**INT7 is the format to build around.** Measured on these weights: 1.304 percent
error at 6.651 coded bits against FP8's 2.57 percent at 8.06. Fewer bytes AND
half the error. It is also the widest code that dequantises into BF16 for free —
BF16 has exactly seven mantissa bits, so an eight-bit code overflows into the
exponent and comes out doubled. `tests/test_dequant.c` has that as an explicit
rejection.

**The static shared-memory limit is 48 KB, not 128.** `ptxas` enforces it as
`0xc000 max`. The 128 KB is L1/shared per SM; a block's static declaration is
capped at 48. Every kernel uses dynamic shared with a runtime opt-in.

**Geometry is a compile-time contract.** A tile that does not fit, a K extent no
swizzle span divides, a height that is not a whole mma fragment — all
`static_assert`. That is deliberate: the failure lands on a laptop, not on the
ring.

**Nine defects were found by reading, none needed a GPU.** Router logits read and
never computed; dense GEMMs using the MoE group tables; RoPE assuming a fused
layout; the attention projection as one GEMM instead of four; the output
projection missing; the MoE finalize missing; the KV cache never written; dense
layers routed as though they had experts; the DSA selection recomputed every
layer. `tests/test_config_coverage.py` exists because two of those were visible
as constants declared and never used.

## What is unfinished, in the order I would do it

**1. Wire `cache/cache.h` and delete the two legacy cache files.**
723 lines covering what 5,086 do: arena, content-addressed sharing,
copy-on-write, two-tier residency, protection, JIT prefetch. 46 public functions
are reachable from 82 call sites in three files.

Do it one caller at a time and compile between: `api/backend.c` has 9
references, `scheduler/scheduler.c` 23, `api/request.c` 50. Take them in that
order — each step is then verified before the next assumes it. Do NOT
reimplement all 46 signatures at once; that is writing a third cache and hoping
it matches the first two.

**2. Drive speculation.** `inference/kernels/speculate.cuh` has verify and accept
for greedy and sampled. `inference/stage/draft_backend.cu` has the DSpark
drafter. Nothing connects them, and no `llms/` sequence calls either.

`ring/sideband.h`'s `LmTapPlan.consumer_rank` is where drafting placement lives.
The old code assumed the last rank because that is where the logits are; for GLM
5.2 that is wrong, since its three dense layers leave rank 0 with slack.

**3. Finish the two linear-attention models.** `kimi_k3` and `qwen_3_6` have
config and unity but no `layer.cuh`. Every kernel they need exists —
`linear_attn.cuh` has the delta rule and the causal convolution.
`tests/test_config_coverage.py` reports them as "no layer.cuh" rather than as
failures.

**4. Configurable model parameters.** Two known defects:

- `DSV4_KV_BITS 8u` is ambiguous. FP8 and INT8 are both eight bits and decode
  completely differently. The cache needs a FORMAT, not a width, and the format
  traits already exist — it should be `DSV4_KV_FORMAT LmInt8`.
- Which ranks draft is a model property with nowhere to live. It belongs in the
  model description alongside `LmTapPlan`.

**5. The DSA index norm.** `GLM52_DSA_INDEX_EPSILON` is exempted in
`test_config_coverage.py` with a reason: I have not read what it normalises and
guessing would be a tenth defect.

## Things that will bite

**`tools/gates.sh` is the contract.** Every check runs through a wrapper that
returns non-zero. A gate that cannot fail is not a gate — an earlier version
reported nine passing assertions from a translation unit that had failed to
compile.

**I deleted things the build needed, twice**, and every gate stayed green both
times, because no gate compiled the tree that used them. Both were in
`model-families/common`, which held a kernel library AND a transport, a
tokenizer, a KV store. A directory named for a property collects whatever shares
the property.

**Regex surgery on C broke things four times** — unbalanced braces, a mangled
build script, invented struct fields, captured call sites. `gcc -aux-info` emits
correct prototypes and is the right tool when a split is genuinely needed.

**`api/request.c` does not split.** 171 functions, 75 cross any boundary I could
draw. It looks like a request lifecycle and an admission scheduler bolted
together and it measures as one thing. The win there is deleting the seven
functions that duplicate `cache/cache.h`, not moving the file.

## Numbers worth not losing

```
GB10 BF16 ridge                573 FLOP/byte
decode AI, B128 to B1024       8 to 64
INT7 unpack cost               ~2 percent of CUDA cores
transport software floor       29 us/hop   (from a deleted probe; see
                                            docs/PROBES_REMOVED.md)
MTP at B1                      3.47 tok/s against plain 3.89 — LOSING
1 TB cache, GLM 5.2 slot       14,904,710 blocks
40 GB device resident          582,542 — 3.9 percent
```

That last pair is why the cache is two-tier. My first version was not, and the
audit before deleting the old one is the only reason it is now.

## Register usage, from the compiler

```
TILE_M=16    52-53 registers, 0 spill
TILE_M=32    63-72
TILE_M=64    83-114
attention    26-29, 2,336 bytes shared
delta rule   76, 1,024 bytes shared
```

## What has never run

None of this has executed. It compiles for `sm_121a`, every mapping is verified
against CUTLASS's own layout files, every code path is checked against an
independent host reference — and no kernel has processed a real weight.

`tests/reference.h` is the oracle for the first run: it shares no code with
`inference/kernels/`, which is the property that makes a disagreement mean
something.

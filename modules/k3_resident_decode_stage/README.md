# k3_resident_decode_stage

A proposed resident decode-stage driver for **Kimi K3** (2.8T MoE), built to the
same module ABI as `glm52_resident_decode_stage`.

K3's weights and technical report are expected **2026-07-27**. This module was
written before that, from what Moonshot has disclosed plus a set of explicit
guesses. Every constant in `include/sparkpipe/spark_k3_model.h` is tagged
`DISCLOSED` or `GUESS` inline; `DIFFERENCES.md` lists each guess with the single
line that changes when the real number arrives.

**Read `STATUS.md` before trusting anything here.** Short version: it compiles
clean, the datapath algebra is CPU-validated, and no kernel has ever run.

## What K3 is, as a driver problem

| | |
|---|---|
| attention | 3:1 **Kimi Delta Attention** to gated **NoPE MLA** |
| residuals | **Block Attention Residuals** — softmax mixture over a stack of block outputs |
| mlp | **Stable LatentMoE**, 16 of 896 experts + shared, **SiTU** = σ(gate)·tanh(up) |
| weights | **MXFP4** E2M1 + E8M0 per-32 scale |
| per-sequence state | KDA recurrent state (fixed size) **and** MLA latent cache (grows) |

The three that shape the module: KDA means state is a *fold*, not a cache
(no prefix sharing, chunked ordered prefill); AttnRes means the pipeline
transport payload is a *stack* that grows along the pipeline (which is why v1 is
single-node); MXFP4 means one decode helper, not NVFP4's two-level scales.

## Layout

```
include/sparkpipe/spark_k3_resident_decode_stage_firmware.h   ABI, weights views, node/frame context, 15 launchers
source/spark_k3_resident_decode_stage_cuda.cu                 19 kernels + launchers
source/spark_k3_stagepack_format.h                            .k3sp wire format + the single-source tensor shape table
source/spark_k3_resident_decode_stage_module.c                host module: load, validate, allocate, layer walk, frame path
tools/k3_pack_synthesize.c                                    synthetic pack writer (--dry-run reports the footprint)
validation/spark_k3_reference.c                               cpu fp32 oracles
```

`../../include/sparkpipe/spark_k3_model.h` holds the model constants and the
provenance tags. The shape table in `spark_k3_stagepack_format.h` is the single
source the loader validates against, the synthesizer emits from, and the cpu
reference reads — a shape cannot disagree between them.

## The layer walk

`RunStage` → `RunLayer` × 72 → `RunAttention` / `RunMlp` → launchers. Per layer:

```
AttnResMix(attention site, completed+1 candidates)
RmsNorm(attention_norm)
KDA (Materialize → DecodeStep | Chunk → Finish)  or  MLA (Decode | Prefill)
AttnResAccumulate(opens_block, completed)
AttnResMix(mlp site, completed+opens+1 candidates)
RmsNorm(mlp_norm)
DenseMlp  or  MoeRoute → MoeExperts
AttnResAccumulate(0, completed+opens)
```

then `AttnResMix(final site, all candidates)` → `RestrictedLogits`.

The running partial is not a separate buffer — it is the last live slot in the
representation array, so opening a block costs no copy.

## Build

```
make -C modules/k3_resident_decode_stage archive     # host + cuda, sm_121a enforced
make -C modules/k3_resident_decode_stage reference   # cpu oracles, no device needed
make -C modules/k3_resident_decode_stage pack        # synthetic stage pack
make -C modules/k3_resident_decode_stage publish     # into the module library
```

`CUDA_ARCH` is checked, not defaulted: the KDA kernel's shared-memory plan is
proven against the sm_121 opt-in cap, so another arch is an error.

At the real geometry a synthetic pack is **1398 GiB** (`--dry-run` reports it
without writing). Use a reduced geometry for anything you intend to actually
load.

## Runtime configuration

Every one of these is **required**. Missing or unparsable is a hard failure, not
a default — a silently defaulted geometry is how a driver ends up quietly
serving the wrong model.

| variable | meaning |
|---|---|
| `K3_STAGE_PACK` | path to the `.k3sp` |
| `K3_MAX_LANES` | concurrent sequences, 1..64 |
| `K3_MAX_CONTEXT_TOKENS` | per-lane context cap, 64..1048576 |
| `K3_PIPELINE_SLOTS` | 1..4 |

## Frame contract

One sequence per frame. `driver_dispatch_slot` must be valid and **is** the lane.

- `buffers[0]`, read, host: `new_token_count` token ids, each `< 163840` —
  an out-of-vocab id is rejected on the host before anything is uploaded
- `buffers[1]`, write, host: `new_token_count` sampled ids
- decode: `new_token_count == 1`; prefill: ≤ 64 (the KDA chunk width)
- `user_context` may optionally override the MLA block table; the table must
  carry a nonzero `lane_stride` and both host mirrors, and its per-lane block
  count is proven to cover the frame's final context position before launch;
  any hidden transport flag is rejected
- `Execute` is thread-safe: each concurrent call claims its own pipeline slot
  (host staging + stream) by compare-and-swap; a saturated node answers
  `SPARK_STATUS_BUSY`. One in-flight frame per lane remains the scheduler's
  invariant, as in glm52.

This diverges from glm52 (which takes `buffer_count == 0` and batches resident
sequences) on purpose — see `DIFFERENCES.md` §7 — and it costs throughput: 896
experts activated for one token at a time.

## Correctness posture

The stage pack header restates the full geometry and is compared field-by-field
against the compiled constants at load; a mismatch names the offending field and
fails. Every tensor is checked for kind, layer scope, format, exact extents and
in-file bounds. Every tensor the layer walk will dereference must have arrived
before any launch. No `ifdef`, no fallback, no silent default anywhere in the
load path: if the pack does not match the model this driver was compiled for, it
refuses to run.

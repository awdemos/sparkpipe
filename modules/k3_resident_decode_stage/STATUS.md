# STATUS — k3_resident_decode_stage

Written 2026-07-17, production-standards audit applied 2026-07-18. K3 weights
and the technical report are expected 2026-07-27.

## Audit 2026-07-18

Every line of both translation units re-read from disk; fixes applied and
re-gated. The material findings, all fixed:

- **Prefill pad rows** now run through the whole layer stack at the padded
  width, so they compute deterministic finite garbage instead of dragging
  stale device memory through the mixes into the chunk kernel. The true token
  count reaches only the chunk kernel's masking and the host wire I/O.
- **Concurrency**: per-slot host staging claimed by CAS (glm52's stdatomic
  pattern, no mutexes), atomic counters, per-slot device token count. All
  pipeline slots previously shared one staging set and one device count.
- **Capacity off-by-one**: prefill capacity was checked against the padded
  width, rejecting end-of-context prefills that Admit had accepted; now the
  true count, matching Admit.
- **Wire token ids** validated against the vocabulary on the host; an
  out-of-vocab id was an out-of-bounds device read in the embedding gather.
- **Block-table coverage** proven per frame from the table's host mirror
  before launch; a short caller table could previously walk the attend kernel
  past its block list.
- **Overflow-safe pack bounds** (`offset > file - bytes` form) for payloads,
  scales and the directory region.
- **Silent defaults removed**: the router-bias null fallback in the route
  kernel (bias is a validated-required tensor) and the silent shared-expert
  skip in the experts launcher.
- **Dead state deleted**: `sequence_ids_by_lane` / `sequence_positions_by_lane`
  device arrays (nothing read them; cold detection is host-side) and a
  heap-allocated single int32 that shadowed a stack-sourced async copy.
- **Snapshot arithmetic**: execution failures are counted separately from
  pre-submission rejections, so `active_submission_count` can no longer
  underflow.

**Nothing in this module has ever executed on a GPU.** There was no GB10 in the
environment it was written in. Everything below is separated on exactly that
line.

---

## Compiles, verified

Both translation units, zero warnings, re-verified from disk after every edit:

```
nvcc -std=c++17 -arch=sm_121a -O2 -Xcompiler -Wall,-Wextra \
     -I include -I modules/k3_resident_decode_stage/include \
     -c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu

cc -std=c11 -Wall -Wextra -Werror -O2 \
   -I include -I modules/k3_resident_decode_stage/include \
   -I modules/k3_resident_decode_stage/source -I /usr/local/cuda-13.1/include \
   -c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c
```

nvcc 13.1. The `-Werror` on the host unit is not decorative: it caught the
strict-aliasing pun in the pack synthesizer.

The tools build under the same flags plus `-lm`:
`tools/k3_pack_synthesize.c`, `validation/spark_k3_reference.c`.

## Validated on CPU, verified

`make -C modules/k3_resident_decode_stage reference` — all oracles pass:

| check | result |
|---|---|
| geometry: 72 layers → 54 KDA / 18 MLA = 3:1, 8 blocks opened, final mix 10 candidates vs budget 10 | pass |
| KDA chunk plan ≡ sequential recurrence, cold chunk | out 1.2e-6, state 6.0e-7 |
| KDA chunk plan ≡ recurrence, carried across 3 chunks | out 3.1e-6, state 1.1e-6 |
| KDA padded partial chunk leaves state unmoved (licenses prefill padding) | 3.6e-7 |
| KDA decay saturation: clamp engages at token 16/64, plan diverges | 3.77 — **pinned as expected-invalid** |
| AttnRes softmax weights sum to 1, identity candidate unchanged | pass |
| router top-16 vs full sort agreement, weights sum to 2.5 | pass |
| SiTU bounded \|out\| ≤ 1 and provably ≠ SwiGLU | max gap 34.4 |
| MLA online softmax ≡ two-pass softmax | 4.5e-8 |

The KDA oracle is two independent implementations — a sequential delta-rule
recurrence and the chunkwise plan the wmma kernel is built from — not one
implementation checked against itself.

Inherited, from the rev2 KDA session (a different transcript, same datapath):
the bf16 wmma kernel measured **out ≤ 1.13e-2, state ≤ 2.0e-3** against the
fp32 recurrence in CPU emulation, flat in chunk count. That was emulation of the
kernel's arithmetic, not the kernel running.

## Cross-checked, verified

- `examples/model_descriptions/k3_resident_decode_stage_firmware.json` and
  `model_contracts/k3.json` were diffed field-by-field against the compiled
  constants in `spark_k3_model.h`: **zero mismatches** (`mla_qk_scale` agrees to
  float32; the JSON carries the exact 1/√128).
- The resident footprint is measured, not estimated:
  `k3_pack_synthesize --dry-run` → 1693 tensors, 1,498,024,968,192 MXFP4 bytes,
  **1398.2 GiB** total. At 0.53125 B/param that back-solves to 2.82T params,
  which is the disclosed 2.8T. bf16 would be 5255 GiB.

## Never run — needs a GB10

Everything device-side. Specifically, none of this is known to work:

- all 19 kernels, at any size
- the shared-memory plan (`SparkK3KdaSmemPlan` computes 94976 B against the
  sm_121 opt-in cap of 101376 B — arithmetic, not a successful launch)
- the wmma fragment paths
- stage-pack load of a real >1 TB file (`fseeko` + 64 MB chunked H2D is written
  and compiles; the largest pack actually written here was none)
- the block table, lane recycling, stream overlap, the completion callback
- any accuracy claim about K3 outputs whatsoever

First bring-up on a real node should be, in order: `make pack` at a **reduced**
geometry, `Initialize` → `Snapshot` → `Destroy` with no frames (proves the
loader, the ledger and leak-free teardown), then a single decode frame, then
the KDA chunk kernel against `spark_k3_reference.c`'s recurrence on the device.

## Cannot run on one node, by arithmetic

1398 GiB of weights do not fit a 128 GB GB10. The full-geometry module is not
single-node servable — ~12 nodes minimum at MXFP4, which is consistent with
sparkpipe's PP13 topology. But this v1 module **rejects** pipeline slicing
(`ValidateSliceIsWholeStack` → SCHEMA_ERROR) because AttnRes changes the
transport payload: the stage boundary must carry the whole representation stack,
not one hidden vector, and the stack grows along the pipeline (see
DIFFERENCES.md §5). So today the module is exercisable only at reduced geometry.
That is the honest position: the driver is complete and the topology it needs is
not built yet.

## Guessed, pending 2026-07-27

The architecture is disclosed facts plus reasoned guesses; `spark_k3_model.h`
tags every constant `DISCLOSED` or `GUESS` inline, and DIFFERENCES.md carries
the full ledger with a fix site per guess. The load-bearing ones:

- **the 2.8T identity** (hidden 7168 × moe_inter 2048 × 3 × 896 experts × 71
  routed layers). Every dimension guess is downstream of making this close.
- **MLA is NoPE**. The only guess that is not a one-line fix if wrong.
- **log decay = −softplus(·)** and with it the decay regime the chunk plan is
  valid in. See below.
- **AttnRes block span 9 layers**, giving the disclosed "~8 blocks".

## Known risk, quantified

The KDA chunk plan is exact only while the running log decay stays above
`MIN_LOG_DECAY = -16` within a 64-token chunk — i.e. mean log decay above
**-0.25/token**. Past that the clamp silently changes the model (measured
divergence 3.77 vs the recurrence, because the gram coupling collapses toward
"no decay between distant tokens").

K3's real decay distribution is unknown. If it is aggressive, the fix is the
chunk width or the clamp — **not** the tolerance. Both sides of the boundary are
permanent tests so this cannot rot into a silent accuracy bug.

## Deliberately not implemented

- **MXFP8 activations** (disclosed). Weights are MXFP4; activations stay bf16.
  A throughput optimization with no device to measure on.
- **Quantile Balancing, Per-Head Muon** (disclosed). Train-time only. A driver
  that applied them at inference would be serving a different model.
- **Speculative decode.** Nothing is disclosed about K3 drafting; glm52's
  speculator has no K3 analogue. `draft_token_count` is 0.
- **Cross-sequence batching.** v1 is one sequence per frame; rejected
  explicitly, not mis-served.
- **KDA prefix sharing.** Not possible for the recurrent state; the MLA latent
  cache could be shared and is not.

# techdebt

The diff between README.md and `git HEAD`. Ledger order: an item leaves this
file by landing with a gate, or by being struck from the README. Nothing
leaves silently.

## K3 path to first token

- **K3 stage execute.** The K3 module answers the two validation questions
  (`modules/k3_resident_decode_stage`); it does not yet execute. Layer/slice
  kernels exist and are host-gated (`inference/llms/kimi_k3`); the wiring of
  K3Engine → `SparkServingDecodeDispatch` → stage module is the A4
  acceptance criterion in MODULE_MAP.md and is not done.
- **Pack format decision: K3PK vs stagepack.** One pack format for K3
  weights (MXFP4 experts + bf16 attention) must be chosen and its
  artifact-check taught; blocks the JIT residency path.
- **Drafter forward for K3.** DSpark rides glm's drafter; K3 needs its own
  draft head or an MTP configuration decision.
- **KDA conv1d tap orientation vs fla reference.** If the window taps are
  reversed the failure is SILENT (numerics drift, no crash). Needs the
  cross-check gate against the fla reference before hardware.
- **State pool admission wiring.** `SparkStatePool` exists and is gated;
  the scheduler does not yet price 434 MB/sequence at admission.

## Stage seam completion (A4 remainder)

- **FrameContext payload config-flow.** The common stage header still
  embeds `SparkGlm52DsparkHiddenTapPlan[AUX_LAYER_COUNT]` and a flashinfer
  B12x MoE recipe — 149 budgeted references in
  `spark_resident_decode_stage.h`, 96 in `module.c`. These become opaque
  model payloads (size + alignment through the ABI, contents through the
  model module) or move behind the linker seam like validation did.
- **Flashinfer B12x byte/launch audit.** The production glm MoE path has
  never had the K3-grade treatment (count launches, count bytes, delete
  both). Rescoped from S2; belongs to the stage seam because the audit and
  the split touch the same 1,174 lines of serving_adapter.cu.
- **Firmware test registration.** `test_glm52_resident_decode_stage_firmware.c`
  builds via the module Makefile but is not a gate; register it linking
  module.c + validation.c so the seam is *executed*, not just compiled.

## Performance ledger (open lines)

- **CUDA graph capture (S5).** ~700 GEMM launches/token remain; the
  engine's step shapes are bucketed for exactly this. Hardware-week item.
- **Indirect-A gather-free GEMM (S1).** Kill the route-gather copy by
  letting the GEMM read A through the route index. Design settled,
  implementation blocked on hardware validation (host recorder cannot
  prove a load path).
- **bf16 / device-staged all-reduce.** TP×PP wide layers currently reduce
  through host staging.
- **P1: cross-sequence prefix reuse aliasing validation.** On by default;
  the byte-identical-logits test under shared-prefix aliasing runs at
  bring-up.
- **P2: async release FIFO wire verification.** Fire-and-forget release
  assumes resident FIFO ordering; verify on the wire, not in the comment.
- **Route-log collection deploy.** 24-byte wire format and the
  Bonferroni-corrected analysis exist; the collector runs when Sparks are
  back (August window).
- **MBU measurement column.** Every number in the README performance table
  is bus-model-derived; BANDWIDTH_LEDGER.md gets the measured column at
  bring-up and the README numbers get replaced, not defended.

- **Topology-aware dispatch.** The README's per-deployment TP_g x PP_s
  selection and chunked-prefill interleaving across placements: scheduler
  work; today's placement is fixed at launch.
- **bf16 KDA state.** Halves the 434 MB/sequence slab and doubles ceiling
  batch; blocked on delta-rule numerical-stability validation.

## Structure (README tree vs reality)

- **`speculation/` does not exist yet** — DSpark lives in
  `inference/stage/draft_backend.cu` (2,344 lines, 599 glm refs) and moves
  out when it gains its second model.
- **`quant/` does not exist yet** — packers live in `runtime/pack` and
  `tools/`; the glm-specific py packers leave common `runtime/pack` first
  (B-tier ledger).
- **B-tier DRY**: LmHead/LmDenseMlp template extraction (0.93/0.98
  similarity across 5/4 models), `tools/pack_common.py`, expert_queue
  parameterization (17 hits).

## Upcoming model drivers

- **GLM 5.5**: bring-up on release; expected to inherit the glm52 module
  wholesale with a new geometry header — that expectation is itself a test
  of the seam.
- **DeepSeek V4 GA**: driver at structural parity from the family sweep
  (PR #491); needs GA weights, pack, and its DSA index cache exercised at
  scale.
- **Qwen 3.8**: driver tracking release; dense+MoE hybrid will exercise
  the scheduler's mixed-layer cohort math.

## Long-run soaks

- Mooncake KV tier under multi-day eviction pressure.
- 16-node ring stability at sustained B≥256 (thermals, RDMA retransmit
  behavior at 267 ms step cadence).

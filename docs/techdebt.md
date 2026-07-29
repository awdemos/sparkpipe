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

## Solutions/(codesize^2) - the measured gap and the plan (2026-07-29)

Census: ~90K product lines + 34.7K test lines. DRY-law debt: 3,667
budgeted glm refs across 57 common paths. The end-state is ~300
irreducible refs (dimension tables only). Ranked removal path:

RULE: model drivers are never size cuts. Every potential frontier
open-source target keeps its driver; a driver is a solution, and the
metric's numerator moves first.

0. **The size gate is live** (tests/test_code_size.py): non-test lines,
   ceiling only descends. Current 106.5K. Realistic architecture-stable
   floor ~60K lines (~500K tokens - whole codebase in one large-context
   window, which is the point); the <100K-token ideal implies a different
   product.
1. **The drafter seam - measured, and it is a rewrite, not an extraction**
   (map 2026-07-29): 174 of 176 functions in api/request.c touch
   glm/dspark symbols - 7,059 of 7,205 lines. The surgery re-founds the
   request API on neutral dispatch/verify types with a model payload
   envelope; glm's wire builders become module source behind the linker
   seam. Expected: ~4.5K generic + ~1.8K glm module, net -1K lines,
   -600 refs, and the seam K3's drafter walks through. (~1,450 refs: api/request.c 637,
   draft_backend.cu 600, scheduler/speculation.c 213). DSpark carries
   glm dspark payload TYPES through common request/scheduler paths.
   Same recipe as the validation tier: neutral payload envelope in the
   common ABI (size + alignment), model module owns the contents,
   linker resolves. Erases 40% of all debt and ~2-3K lines of
   glm marshaling.
2. **FrameContext payload config-flow** (149 + 96 refs) - the A4
   remainder below; same envelope mechanism as (1), do together.
3. **W8LUT deletion** (deprecated; 173 refs, 10 files, ~800 lines:
   plan family in firmware header, stagepack branches, checks).
4. **Seam-include tier** (~600 refs: http_server 224, prefix_cache 214,
   scheduler 174 are mostly glm header includes + constants that fall
   out once (1) and (2) land).
5. **api/compat_api.c: 833-line cut candidate.** No OpenAI-style routes;
   sole reference is http_gateway.h. Confirm no external caller, then
   delete whole. (Decision: ct.)
6. Dead complexity: 4 "was deleted" Makefile error stanzas,
   legacy_entry.cu (42 lines), --dspark compat no-op flag.
7. B-tier templates (LmHead 0.93x5, LmDenseMlp 0.98x4, expert_queue
   17 hits, pack_common): ~400-700 lines.

Post-plan: ~85K product lines, debt ~300, same solutions - the metric
roughly doubles. Items (1)+(2) are one A4-part-two-scale surgery.

## Model swap (16x single-tenant by mode)

K3 (1.6 TB) + GLM-class (~0.4 TB) cannot be co-resident with useful KV
(2.0 TB vs 2.0 TB capacity). Swap = drain + parallel NVMe load of ~100
GB/node: 14-20 s at 5-7 GB/s local NVMe, plus arena re-init. Target
<20 s via pack mmap + JIT residency warm path; orchestration (drain,
load, announce) is scheduler work. Big-batch workloads run the smaller
resident model; K3 owns B1-B8 interactive with shared-prefix reuse.

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
- **Consecutive-token routing overlap (DSpark at B=1).** If adjacent
  positions route to overlapping experts, verify rows share weight reads
  and B=1 speculation turns real (~x1.5 hypothesis). Measure from route
  logs at bring-up.
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

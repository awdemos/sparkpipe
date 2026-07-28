# Module map

Every root, its size, its tier, and its DRY verdict. Sizes are line counts
from the 2026-07-28 survey (C = .c/.h, CUDA = .cu/.cuh); re-run the survey in
docs/DRY_LEDGER.md's spirit before trusting numbers over code. Tiers:

  COMMON   - no model name, under the law gate (tests/test_dry_law.py)
  MODEL    - one model's data and kernels; the only places a model name lives
  SEAM     - generic machinery still fused to glm data; named in the ledger
  TOOLS    - build/pack/analysis; TESTS - the gates; DOCS/DATA

| module | files | C | CUDA | py | tier | role and verdict |
|---|---|---|---|---|---|---|
| tests/ | 91 | 25,334 | - | 5,950 | TESTS | the 45 gates + fixtures + host_cuda harnesses/shims. Per-model host shims are small and earn their keep |
| api/ (+gateway) | 6 | 14,522 | - | - | COMMON | http gateway, request, compat - impl of the spark_http_gateway/request headers. 1,334 glm refs, all seam-budgeted |
| node/ | 5 | 12,215 | - | - | COMMON | the four daemons + memlink tool. Renamed in A1; 139 budgeted seam refs |
| docs/ | 93 | - | - | - | DOCS | 11.9K lines incl. DRY_LEDGER.md and this file |
| inference/stage/ | 6 | 4,983 | 4,015 | - | SEAM | **A4, the biggest un-split seam**: the stage-module framework (module.c, dispatch, runner) fused with the glm serving adapter (serving_adapter.cu, draft_backend.cu, legacy_entry.cu) - 1,953 glm refs. The split is framework vs per-model adapter, and it is exactly where K3's backend adapter plugs in. Compiled by modules/glm52_resident_decode_stage/Makefile |
| ring/ (+transport) | 6 | 2,839 | 5,062 | - | COMMON | tcp, rdma, memlink, hidden transport, tp_collective (the all-reduce), sideband. Zero model refs |
| runtime/pack/ | 7 | 6,439 | - | 1,007 | SEAM | stagepack.c (the C pack loader) + driver compiler, module library, model description, artifact check - plus **two glm py packers misplaced in a common dir** (stage_pack.py, fp8_resident_pack.py -> move to tools/, glm-named). See "the four pack surfaces" below |
| scheduler/ | 5 | 7,177 | - | - | COMMON | impl of scheduler, long_context, stage_plan, speculation headers. 645 budgeted |
| include/sparkpipe/ | 36 | 6,655 | - | - | COMMON | the common headers, post-A1 |
| tools/ | 34 | 2,247 | - | 3,293 | TOOLS | packers, shard, param budgets, dspark manifests. k3_pack/k3_shard gated; glm52_* py tools glm-named correctly |
| cache/ (+store) | 5 | 6,086 | - | - | COMMON | kv_cache.c, prefix_cache.c, stage_kv_client - impl of the cache headers. 665 budgeted; the kv GEOMETRY seam (A3) lives here + model-families/glm52/kv_cache.h |
| inference/llms/ | 25 | 1,497 | 3,462 | - | MODEL | five model dirs. K3 the largest and only slice-extracted one |
| inference/kernels/ | 25 | - | 4,762 | - | COMMON | the Lm kernel library: gemm, tile, formats, attn, linear_attn, norm, route, topk, project. Zero model names |
| text/ | 5 | 4,047 | - | - | COMMON | chat template engine, prompt, tokenize. 140 budgeted (template DATA is glm's) |
| modules/glm52_resident_decode_stage | 11 | 2,812 | 225 | - | MODEL | glm stage firmware module + its Makefile (which compiles inference/stage) |
| runtime/ (rest) | 8 | 2,776 | 255 | - | COMMON | launch planning, tensor maps, gemm launchers |
| model-families/glm52/ | 15 | 2,230 | - | - | MODEL/SEAM | post-A1 remainder: kv_cache.h (A3), tp_shard (A2), stagepack tables, dspark, model.h, shape/chat/text data, expert_queue |
| src/ | 5 | 1,483 | - | - | COMMON | driver loader, orchestrator, sha256, filesystem. Zero model refs |
| serving/ | 6 | 1,363 | - | - | COMMON | A1's six moved sources |
| deployment/ | 3 | 2,928 | - | - | COMMON | deploy tool + src; 14 budgeted |
| examples/, schema/, model_contracts/ | 13 | - | - | - | DATA | model descriptions, release notes, contracts |
| (root) | 7 | - | - | - | DATA/BUILD | Makefile, sources.mk, locks/specs (~26K, mostly generated data) |

Totals: ~66K C, ~18K CUDA, ~10K py, 45 gates.

## The four pack surfaces (one format decision owed)

1. `runtime/pack/stagepack.c` - the C loader residentd speaks, glm's format
2. `runtime/pack/stage_pack.py` + `tools/glm52_resident_pack_common.py` - glm packers
3. `tools/k3_pack.py` - the K3PK manifest format, gated, torch-free
4. `tools/k3_shard.py` - pack-side TP slicing of (3)

Two resident formats now exist. The decision - K3 loader learns K3PK, or
k3_pack emits stagepack - is deferred to hardware week deliberately: it needs
residentd's mmap path in front of it. Whichever loses, its gate transfers.
Until then the law gate counts stagepack references as declared seam debt.

## Law-gate scope (this commit)

COMMON grew from 4 roots to 12: api, cache, scheduler, text, src, runtime,
deployment and inference/stage joined include/sparkpipe, node, ring, serving.
Budget: 54 files, 5,151 references, every one classified to its seam. Two
outright law violations were found and fixed on arrival - post-law model
names in comments in runtime/launch.h and cache/store/stage_kv_client.c -
which is the gate doing its job on day one.

## Standing DRY items, consolidated (ledger order)

A2 tp-shard split; A3 kv geometry seam (cache/ + glm kv_cache.h);
**A4 inference/stage framework/adapter split** (new, the K3 backend's
doorway); pack format decision; move the two glm py packers out of
runtime/pack; B-tier: LmHead (0.93 x5) and LmDenseMlp (0.98 x4) extraction,
pack_common.py, expert_queue parameterisation.

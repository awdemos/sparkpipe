# Handoff: KV sharing, kernel work, and what not to re-derive

Written at the end of a session that spent a lot of budget re-deriving things
and making three wrong calls from incomplete searches. Read the METHOD section
first; it is the part that would have saved the most time.

## METHOD - the failures worth not repeating

1. **Grep for CALLERS, not for files.** A component existing, being tested, and
   being in `sources.mk` says nothing about whether the runtime uses it. I
   concluded "JIT KV caching was never wired" because `jit_kv_pool.c` had no
   production callers. That was wrong: production has JIT KV caching, in a
   different file (`work_control.c`). The conclusion was drawn from one grep.
2. **Never infer behaviour from a name.** I claimed `physical_block_pin_counts`
   "is already the refcount" and that eviction "already does LRU". Reading found
   pins ARE a refcount but there is a `!= 1u` hard constraint in one path (it is
   scoped to transient scratch blocks, so it does not block sharing - but the
   name gave no way to know that), and eviction is a CLOCK SWEEP with an epoch
   guard, not LRU.
3. **Search the repo before implementing.** I wrote content-hash + refcount
   sharing into the JIT pool without finding `spark_glm52_kv_dedup.c`, which
   already implements exactly that, better. The commit was reverted.
4. **The repo has a large measurement corpus.** `diagnostics/` holds 2533 files
   across 18 campaigns. Read it before claiming data does not exist.

## ENVIRONMENT (verified)

- No GPU, no CUDA runtime. `nvcc 13.1` compiles only; **nothing in this
  container has ever executed a kernel**. Host C tests DO run.
- No cuBLAS SDK, so `spark_glm52_pp13_node_context_builder_cuda.cu` and
  `spark_glm52_sm121_required_decode_stage.cu` have **no local compile gate**.
  Changes there are diff-review only.
- Target arch `sm_121a`. Probed: **TMA (`cp.async.bulk.tensor`) and
  `__cluster_dims__` ARE available; `tcgen05` is NOT.**
- Repo was restructured (PRs #505/#506): shared kernels now live at
  `model-families/common/include/sparkpipe/`, model headers at
  `model-families/<family>/include/sparkpipe/`, but `.cu` sources are still
  under `modules/<family>_resident_decode_stage/source/`.
- Compile a family: `nvcc -std=c++17 -arch=sm_121a -O2 -Xcompiler -Wall,-Wextra
  -include model-families/<f>/include/sparkpipe/spark_<f>_model.h
  -I model-families/common/include -I model-families/<f>/include -I include
  -I modules/<f>_resident_decode_stage/include -c <source.cu> -o /tmp/x.o`
- `tools/length_gate.py <files>` enforces the 50-line rule and **exits nonzero**
  (it used to only print, which let four violations ride through a `&&` chain).

## TARGETS (from the owner, not inferred)

- **No precision loss.** FP4 weight quantization is rejected. Anything that
  converts weights to a narrower type for speed is off the table. FP8 weights
  with FP32 accumulate are fine because the weights already ship as FP8.
- **MLA is the only attention path.**
- **B8 is the default chat batch**, submitted by centaur as **shared-prefix**
  requests. **B256-B1024** for code audits and benchmark runs. B1 and B128 are
  NOT the targets - earlier tuning aimed at B1 and should be re-judged.
- Deployment: **TP13 for chat, PP13 for batch**, possibly mixed, with a **30
  second or less mode switch** and pending requests suspended across it.

## KV ARCHITECTURE - GROUND TRUTH (read, not inferred)

The production KV manager is `SparkGlm52Pp13WorkControlKvState` in
`model-families/glm52/src/spark_glm52_pp13_work_control.c`. It already has:

- An **open-addressed directory** (`KvDirectoryFind`, `KvDirectoryHash`, linear
  probing, tombstones) keyed by **`(sequence_id, logical_block_index)`**.
- **`physical_block_pin_counts`** with a real Pin/Unpin refcount (increment
  guards `UINT32_MAX`, decrement guards 0). A separate
  `ReleaseTransientPhysicalBlock` requires `pin == 1`, but only for TRANSIENT
  scratch blocks (`sequence_id == UINT64_MAX`), so it does not constrain sharing.
- **Eviction: a CLOCK SWEEP** over `next_physical_block_index`, which skips
  blocks with `pin_count != 0`, skips blocks touched in the current `epoch`, and
  requires `RESIDENT` + backing capacity to swap out.
- **NVMe paging, wired**: `backing_block_*` free list, `swap_store_function` /
  `swap_load_function`, `swap_store_count`, `swap_load_count`,
  `swapped_block_count`, `clean_evict_count`.

**The only gap is the directory key.** Because it is keyed by sequence, two
sequences with an identical prefix get two physical blocks holding identical
bytes. For B8 shared-prefix chat that is 8 copies stored and 8 read per step.

## THE DUPLICATION PROBLEM

| file | residency + paging | content dedup | reachable from runtime |
| --- | --- | --- | --- |
| `work_control.c` KvState | yes, full | **no** | **yes** |
| `glm52/src/spark_glm52_jit_kv_pool.c` | yes, duplicate | no | no |
| `glm52/src/spark_glm52_kv_dedup.c` | no | yes | no |

`jit_kv_pool.c` reimplements production residency for the simulator.
`kv_dedup.c` implements the one thing production lacks. Both are compiled into
the shipping driver via `sources.mk` and called only by
`tests/test_glm52_batch_plane.c` and `tools/sparkpipe_glm52_batchplane_sim.c`.

This is also why the batch-plane simulator's predictions never matched reality:
it faithfully models components the runtime does not run.

## DESIGN - prefix sharing, minimal and DRY

1. Add a **content -> physical index for SEALED blocks only**, reusing
   `KvDirectoryHash` and the existing probe/tombstone pattern. Do not write a
   third probe loop; there are already two in that file.
2. **Keep the `(sequence, logical)` directory.** A sequence must still find its
   own blocks. Sharing is many-to-one from that directory into physical blocks.
3. On **seal**, register `content_hash -> physical_block_index`.
4. On **acquire**, look up content first. On hit, point the new directory entry
   at the existing physical block and **Pin** it. On miss, allocate as today.
5. **Eviction needs no change** - the existing pin guard protects shared blocks,
   and clock-sweep-plus-pins is more robust for a hot prefix than recency.
6. **INVARIANT, the one silent-failure risk:** only fully-written (sealed) blocks
   may be shared. A partially-filled tail block must stay private, or one
   sequence's continuation corrupts another sequence's prefix.
7. Then **collapse the duplicates**: fold `kv_dedup.c` into work_control, delete
   `jit_kv_pool.c`, and rebuild the simulator on `WorkControlKvState`. One
   residency implementation, one identity implementation, both the ones
   production runs.

`tests/test_glm52_shared_prefix_admission.c` (on PR #507) pins the composition
contract and RUNS: 8 rows, one prefix hash, 1 physical admission, 7 shares.

## WHY MLA MAKES THE MODE SWITCH WORK

MLA stores one **head-independent** latent row per (token, layer). So the latent
cache is **replicated across TP ranks, not sharded**, and a content-hashed
fragment is byte-identical on every rank and across a PP relayout.

- **TP -> PP is a pure discard** (each rank already holds all layers; keep your
  slice, free the rest, zero network).
- **PP -> TP must NOT gather in memory.** At `kv_pool_token_capacity = 1376256`
  and 1152 B/token/layer, ~11 GB/rank; gathering 13 slices is ~144 GB against
  **775 MB free on rank 12 at B256** (`diagnostics/glm52_b256_compressed_mla_20260715`).
  It must reload through the KV store instead.
- Spilling ~11 GB/rank at the documented ~6 GB/s NVMe budget is ~2 s, ranks in
  parallel - comfortably inside 30 s. Sharing makes it cheaper still.
- **Verify `layout_fp` covers TP/PP geometry** before trusting a cross-mode cache
  hit. MLA's geometry-independence should make it safe; the failure mode if it
  is not is silent wrong context, not a crash.

## MEASUREMENT REALITY

`diagnostics/` granularity is **per-stage** (`timings.tsv`: first_layer,
layer_count, total_us, maximum_us, graph_replays) and **per-request**
(`*.summary.json`: latency, TTFT, `token_events_per_s`). There is **no
per-kernel breakdown anywhere** - no `.nsys-rep`, no kernel-name timing table.
So "which kernel owns the time" is still not answerable from the repo. That one
nsys capture remains the highest-value datum from ring access.

Calibration notes live in `docs/GB10_CUDA_COST_MODEL_CALIBRATION.md`; the model
is in `tools/sparkpipe_family_cost_model.c`. Both are projections from the
measured stage buckets, not silicon truth for the family drivers, none of which
has ever run.

## PR #507 - what is already done

Branch `agent/tile-k-alignment-guard`, base `main`. All compile-clean; only the
host C test executes.

- Hard-fail on non-K-aligned width and on unhandled weight format in the linear
  dispatch (both were silent wrong-number paths; F32/U32 fell through to the FP8
  decoder and were read as packed E4M3 bytes).
- FP8 tile: per-K-tile activation scale, removing a full-K rescan that was 272x
  redundant at worst and could make FP8 slower than the bf16 tile it replaces.
  Finer granularity is also strictly more accurate.
- Gate/router input staged once per row instead of once per expert block:
  33.6 MB -> 1.0 MB per layer at B128, bit-identical numerics.
- Attention head grouping chosen by batch instead of pinned at 4.
  **Re-judge this against B8, not B1** - it was tuned before the targets were known.
- `tests/test_glm52_shared_prefix_admission.c`.

## NEXT, in order

1. Prefix sharing in `work_control.c` per the design above. Highest value; serves
   the stated primary workload; reduces memory pressure at B256+.
2. Collapse the duplicate components and rebuild the simulator on the production
   KV state.
3. Spill/restore through the existing KV store, then TP<->PP reconfiguration.
4. Re-judge attention grouping and the tile M-chunking for B8 and B256-B1024,
   which are the real targets and neither of which has been modelled.

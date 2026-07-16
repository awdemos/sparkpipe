# GLM-5.2 Implemented But Inactive Audit

## Scope

This audit replaces the stale ten-item list in
`GLM52_FP8_SCALED_GEMM_ACTIVATION_20260712.md`. It reviews source at
`7c01406578286930c1444f8ee06a7aa0e8347ecd` and the live FP8 B256 MTP release
observed on 2026-07-16:

```text
glm52-fp8-main-7c014065-b256-mtp-priority-abi19
```

An item is called active only when there is source reachability plus a live
receipt. Compiled code, a capability bit, a test, or a release argument that
was not selected is not evidence of activation.

Management access to the Spark ring failed after the live process, health,
and benchmark observations were collected. The source findings below are
current to the audited commit. Live findings are limited to the observations
already captured in this session and are not extrapolated past the route loss.

## Review Of The Old Ten

| Old item | Current verdict | Evidence |
| --- | --- | --- |
| Production batching and queuing | Active, with a prefill failure | The live 184-request run selected width 15 and honored priority. A 64-token prefill wave still monopolized progress for 182 seconds and disconnected resident IPC. |
| Actual SSE streaming | Active | A normal smoke request streamed a token before completion. The 184-request failure stranded final events, which is a correctness bug rather than buffered-only SSE. |
| Native FP8 Q/KV/O scaled GEMM | Active and fail-closed | Live qualification reported 54 of 54 scaled GEMM plans bound. The FP8 launch returns `MODULE_NOT_VALIDATED` when its backend is absent; the BF16 WMMA branch remains only for the FP4 plan kinds. |
| Bulk/paged prefill | Partially active | Prefill is batched across sequences at a shared token offset. It is still token-serial: the builder plan sets `maximum_prompt_token_count = 1`, and long prompts become one ring wave per token. |
| Q/KV branch overlap | Inactive by construction | The builder allocates streams and events, then explicitly removes `QKV_BRANCH_OVERLAP` from the exact-stage capability set. |
| RDMA hidden transport | Inactive in the release | RDMA and persistent-doorbell code exists. The live manifest and process select `libhidden_transport_tcp_cuda.so`. |
| JIT KV and DSA fragment transport | Inactive in the release | Live JIT counters were zero and resident arguments omitted all NVMe options. The builder also explicitly clears the DSA fragment-transport flag and both plans. |
| MTP | Active | Live health reported draft, verify, accepted, committed, and rejected counters. Chained verification is merged and measured. |
| FP8 KV cache | Unreachable from the PP13 builder | Lower CUDA contracts and kernels exist, but the builder never creates or assigns `fp8_kv_cache_plan` and never requires the FP8 KV execution flag. |
| Large batch buckets | Active through B256 only | The live release is B256 and selected dynamic widths below that capacity. B512 and B1024 launchers exist but were not part of the live release. |

The old list also omitted two now-proven active paths: the exact six-layer PP13
stage launcher and the event-driven resident work scheduler.

## Current Inactive Code

This is the replacement list, ordered by expected impact on the measured
service rather than by implementation size.

### 1. Multi-token bulk prefill

The paged and bulk prefill contracts exist, but the PP13 builder constructs a
one-token plan:

```c
layer->serial_prefill_paged_plan.maximum_prompt_token_count = 1u;
layer->serial_prefill_bulk_plan.maximum_prompt_token_count = 1u;
```

The current sequence batching only makes one token position wide across
requests. It does not process a prompt chunk through the ring as a unit. The
live 64-token waves taking 7.4 to 8.5 seconds, plus one 182-second stalled
wave, are the activation receipt that is still missing.

### 2. NVMe-backed JIT KV cache

The resident has complete configuration, storage records, batched load/store
queues, qualification counters, and CLI options. The live resident command
contains none of `--kv-nvme-path`, `--kv-nvme-blocks`, or
`--kv-nvme-batch-blocks`; health reported zero prefetch starts, completions,
and blocks. The code is built but does no work.

### 3. DSA KV-fragment transport

The required CUDA module contains fragment prefetch/save plans and launch
logic. The production PP13 builder disables it explicitly after layer
initialization:

```c
layer->node.reserved_execution_flags &=
    ~SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_DSA_KV_FRAGMENT_TRANSPORT;
layer->node.dsa_kv_fragment_prefetch_plan = 0;
layer->node.dsa_kv_fragment_save_plan = 0;
```

DSA score selection is not the same feature. The absent fragment transport is
the part intended to fetch selected long-context KV from outside the resident
GPU pool.

### 4. FP8 KV cache

The required CUDA module implements FP8 E4M3 MLA, key-nope, and value cache
stores and reads with dynamic scales. No non-test source outside that module
assigns `node_context->fp8_kv_cache_plan`. The PP13 builder continues to bind
BF16 cache storage, so the FP8 cache implementation is unreachable in the
deployed driver.

### 5. Q/KV projection overlap

The builder creates `query_stream`, `kv_stream`, three events, and fills the
exact-plan stream/event fields. It then masks out the only capability that
allows the stage launcher to overlap the branches. These allocations and
teardown paths remain live overhead around an inactive execution mode.

### 6. GPUDirect RDMA persistent-doorbell transport

The RDMA module and persistent doorbell fast path are merged. The release
selects the TCP CUDA transport instead. No live RDMA latency or throughput
receipt exists for the inference path at this commit.

### 7. Memlink multi-lane RAM service

`sparkpipe_memlink`, `prevcp`, `nextcp`, the multi-lane daemon, and reusable C
neighbor helpers all build. Nothing in the PP13 hidden transport, resident,
rank daemon, gateway, or release manifest calls the memlink API. It remains a
standalone transfer tool rather than part of inference.

### 8. DSpark speculative decoding

The draft backend, epoch-3 checkpoint validator, tap transport, request API,
resident IPC, and gateway switches are merged. The live release omits
`--dspark`, no DSpark checkpoint is resident, and health reports
`dspark_status=NOT_WORKING`. This is a complete inactive stack, not a measured
speed feature.

### 9. B512 and B1024 execution buckets

The exact PP13 launcher table contains B512 and B1024 entries and the resident
defaults permit 1024. The live release is capped at B256. These paths therefore
have source and compile coverage but no current ring execution or accuracy
receipt.

### 10. Continuous release-manager agents

The release manager implements persistent agents and restart-on-change roles.
No persistent agent process was running on the inspected ring. The gateway had
PPID 1 and recovery required a manual release-manager `agent --once` launch.
The supervision code therefore was not supervising the live service.

### 11. W8LUT runtime driver

The isolated W8LUT pack format, conversion watcher, resident MoE runtime,
BF16 trunk packs, qualification gates, and release mode are merged. The live
driver is FP8. W8LUT is intentionally a separate inactive driver while its
stagepacks are generated; it must not be described as a live quality gain.

### 12. Prefix-family reuse observability

Prefix-family selection and counters exist in the request API and serving
engine. The gateway health output used by the live benchmark does not expose
those counters, so there is no receipt proving that the duplicated DS4 prompts
shared prefill work. This item is classified as unproven rather than inactive:
the scheduler can select it automatically, but the service currently cannot
show that it did.

## Source Evidence Index

| Finding | Current source anchor |
| --- | --- |
| One-token prefill plans | `modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu:2773` and `:2815` |
| DSA fragment transport disabled | `modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu:4227` |
| Q/KV overlap disabled | `modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu:4268` |
| FP8 KV has lower-level consumers only | `modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu:17731` |
| B512/B1024 launcher entries | `modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu:22358` |
| NVMe resident switches | `tools/sparkpipe_glm52_cuda_residentd.c:281` |
| DSpark gateway switch | `tools/sparkpipe_glm52_http_gateway.c:478` |
| Prefix-family counters | `src/spark_glm52_request_api.c:3716` |
| Memlink remains a standalone tool | `tools/sparkpipe_memlink.c:1241` |

## Activation Order

1. Make multi-token prefill asynchronous and chunked without monopolizing the
   gateway or resident IPC.
2. Expose prefix-family counters and prove shared-prefix reuse on the paired
   DS4 workload.
3. Activate and qualify NVMe JIT KV together with DSA fragment transport for
   long-context capacity.
4. Activate Q/KV overlap only after a device-timed equivalence gate; remove its
   unused streams and events if it loses.
5. Measure RDMA against TCP in the same ring check and inference workload, then
   keep one transport.
6. Bind FP8 KV only if its capacity or bandwidth result beats compressed BF16
   MLA at equal accuracy; otherwise delete the unreachable implementation.
7. Treat DSpark, W8LUT, and B512/B1024 as separate qualification efforts, not
   implied capabilities of the current FP8 B256 service.

## Reporting Rule

Future status must use one of these labels:

```text
ACTIVE_MEASURED
ACTIVE_UNMEASURED
BUILT_NOT_SELECTED
UNREACHABLE
NOT_BUILT
BROKEN
```

`Implemented`, `connected`, `supported`, and `production` are not runtime
status values.

# GLM-5.2 Runtime Activation Audit

## Status Rules

Runtime status uses only these labels:

```text
ACTIVE_MEASURED
ACTIVE_UNMEASURED
BUILT_NOT_SELECTED
UNREACHABLE
NOT_BUILT
BROKEN
```

This audit describes the seven-path integration based on source reachability and
host tests. It makes no Spark-ring speed claim. SM121 compilation, accuracy, and
performance remain hardware gates.

## Integrated Paths

| Path | Status | Current source behavior | Hardware receipt required |
| --- | --- | --- | --- |
| Multi-token bulk prefill | ACTIVE_UNMEASURED | Work ABI 10 carries row-major prompt tokens and ragged row counts. The service chunks up to 256 prompt tokens and the builder launches one embedding gather for the execution rows. | Prompt accuracy, chunk latency, and ring progress at 64 and 256 tokens. |
| NVMe JIT KV | ACTIVE_UNMEASURED | The release supplies a stable `.jit` path, 1,048,576 records, 128-record I/O batches, a separate nonblocking CUDA stream, and FP8-aware record ABI 3. | Forced eviction and reload with nonzero completion counters and no stale records. |
| DSA KV fragment path | ACTIVE_UNMEASURED | DSA selection now builds 32 logical blocks from 2,048 selected tokens and issues read-only L2 prefetches for resident blocks. | Long-context DSA accuracy and stage-clock comparison. This does not yet prove selected-only external-NVMe capacity. |
| FP8 KV | ACTIVE_UNMEASURED | MLA cache bytes and scales remain compressed in device memory and are dequantized directly by attention. Expanded BF16 key/value shadow caches are not allocated in FP8 mode. | Cache byte comparison and end-to-end token parity against the serialized FP8 reference. |
| Q/KV projection overlap | ACTIVE_UNMEASURED | Production plans retain the overlap capability and use the existing query/KV streams and events. | Device-clock proof that overlap beats the sequential path at equal output. |
| Persistent-doorbell RDMA | ACTIVE_UNMEASURED | The generated release selects the verbs transport. Sends use CUDA events and host callbacks instead of synchronizing the caller stream. | Hidden-only and sideband ring laps, then inference hop and stage timing. |
| Memlink multi-lane partitioning | ACTIVE_UNMEASURED | The RDMA transport uses the shared memlink partition helper for lane striping. The standalone RAM object service is not inserted into inference. | Per-lane byte balance and aggregate fabric bandwidth. |

## Important Boundaries

The DSA integration is a resident read-only L2 prefetch. The current work-control
JIT path still loads context blocks before DSA selection, so selected-fragment
external-NVMe capacity is not claimed.

The memlink integration reuses its multi-lane partition contract inside RDMA.
Inference does not route hidden states through the standalone memlink daemon.

No compatibility fallback was added. Missing required capabilities, malformed
bulk token rows, invalid FP8 cache plans, and unavailable RDMA dependencies fail
closed.

## Remaining Inactive Or Unproven Code

| Item | Status | Reason |
| --- | --- | --- |
| DSpark | BUILT_NOT_SELECTED | Separate qualification effort; not part of this seven-path release. |
| W8LUT | BUILT_NOT_SELECTED | Separate model driver and pack format. |
| B512/B1024 | ACTIVE_UNMEASURED | Launchers exist, but this integration does not add a ring receipt. |
| Continuous release agents | ACTIVE_UNMEASURED | Release-manager code exists; live supervision must be verified during deployment. |
| Prefix-family reuse counters in public health | UNREACHABLE | Scheduler counters exist but are not yet exposed by the service health surface. |

## Acceptance Order

1. Build the SM121 archive and resident artifacts from merged `main`.
2. Run hidden-only and 8 KiB sideband ring checks with the RDMA transport.
3. Run the matched real-prompt FP8 accuracy receipt with detailed dumps enabled.
4. Run 64-token and 256-token bulk-prefill receipts.
5. Force JIT eviction and reload and verify the NVMe counters and cache parity.
6. Exercise DSA above 2,048 candidates and compare stage clocks.
7. Compare Q/KV overlap device clocks and keep it only if it wins.
8. Record B1, B4, B16, B64, B256, and B1024 end-to-end measurements where memory permits.

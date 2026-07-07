# GLM52 Device Server Contract

## Invariant

Control-plane changes must never cost a VRAM reload. The reload boundary is
the **process**, not the binary layout: everything dlopened into the residentd
process (the residentd binary, the resident decode stage builder .so, the
hidden transport .so) shares the CUDA context and the ~100GB of resident
state. Moving code between those three binaries changes nothing. A behavior
change avoids reload only if it lives in another process or arrives as data
over the frozen command ABI.

## Process inventory

Device process (restart = full reload, per node):
- `tools/sparkpipe_glm52_cuda_residentd.c` — IPC server, ingest dispatch
  reconstruction (a pure function of the frozen wire structs), transport
  session ownership.
- resident decode stage module .so — node context, weights, KV pools, runner
  threads, kernels, execute step policy.
- hidden transport .so — data plane; it stream-syncs and copies device and
  pinned host staging, so it cannot leave the CUDA process.

Agent processes (restart freely, no reload):
- rank daemon (ranks 1-12) — work queue, control packet forwarding, BUSY
  defer/retry policy, final event send, all evolving scheduling behavior.
- gateway + service backend (rank 0) — API, serving engine, scheduler,
  dispatch marshaling, pending decode matching, final event receive.

Agent-restart freedom is load-bearing and currently holds: residentd unlinks
its stale socket before bind, recycles client slots on disconnect, and (as of
this change set) ignores SIGPIPE so a dying agent cannot kill the device
server.

## Frozen command ABI (version 2)

HELLO 1, HELLO_ACK 2, SUBMIT_WORK 3, SUBMIT_RESULT 4, COMPLETION 5, QUERY 6,
STATS 7, SHUTDOWN 8, ERROR 9, SUBMIT_PREFILL 10, SUBMIT_DECODE 11. The wire
structs in `include/sparkpipe/spark_glm52_cuda_resident_ipc.h` are the device
ABI. The header abi_version gates lockstep; mismatches fail loudly.

Known gaps, to be batched into the next planned device deploy, never shipped
alone: CANCEL/abort-sequence (a stuck live request currently has no remedy
short of device restart) and a config command for execute-policy knobs.
Kinds 12 and 13 are reserved for these.

## Rules

1. A diff touching residentd, the resident decode stage module, or the
   transport module must state in the PR why the behavior cannot be expressed
   as data over the existing ABI or moved to an agent.
2. New policy is agent code or command parameters. Never device code.
3. Device-side changes batch into single reload windows. Agents deploy
   continuously.

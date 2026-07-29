# MODULE_MAP v2 — the canonical modules and their contracts

Twelve modules. Every non-test file belongs to exactly one; a file that
straddles two is a bug in the map or the file. The contract line is the
module's whole job — if a change doesn't fit the line, it goes elsewhere.

| module | contract | files (dirs) | lines |
|---|---|---|---|
| **core** | status codes, hashing, fs, process orchestration, driver dlopen | `src/` | 1.5K |
| **text** | bytes → tokens → chat-templated prompts | `text/` | 4.0K |
| **transport** | move bytes between ranks: RDMA first, TCP fallback, memlink local, TP collectives — all behind `spark_hidden_transport` / `spark_tp_collective` | `ring/` | 7.9K |
| **kv** | paged token arena + geometry-by-request, prefix cache, store tier, Mooncake client, state pool | `cache/`, `spark_state_pool.h` | 6.1K |
| **pack** | weights on disk → validated, addressable artifacts: model description, module library, driver compiler, artifact check, stagepack | `runtime/pack`, `spark_module_abi` | 7.1K |
| **driver** | the model-execution contract and its per-model implementations: Lm kernel family + `inference/llms/*` | `inference/kernels`, `inference/llms` | 9.9K |
| **stage** | the resident decode stage lifecycle, model-agnostic, model linked per family | `inference/stage`, `modules/*` | 9.0K |
| **scheduler** | who runs next and with what budget: cohorts, work control, stage plans, long-context policy, speculation policy | `scheduler/` | 7.2K |
| **request** | request lifetime from admission to emit: slots, rows, MTP trees, sequence tables | `api/request.c`, `serving/spark_row_allocator.c`, `spark_mtp_tree`, `spark_batch_sequence_table` | 8.5K |
| **gateway** | the wire: HTTP+SSE server, OpenAI/Anthropic-compatible bodies, chat-request compat | `api/gateway`, `api/http_gateway.c`, `api/compat_api.c` | 3.3K |
| **node** | the two-process rank runtime: residentd owns CUDA residency across restarts, rank_daemon+backend own the socket→driver pump; IPC between them | `node/`, `spark_cuda_resident_ipc` | 12.2K |
| **serving-glue** | topology, TP shard tables, service backend adapters, node-context building | `serving/` | 1.6K |

Cross-module law: dependencies point downward in this order —
core → text/transport/kv/pack → driver → stage → scheduler → request →
gateway/node. The naming law and the size gate enforce the edges' cost.

## Architecture audits — where the big gains hide (measured, questioned)

1. **`api/serving_engine.c` (2,500 lines): NOT superseded — resolved.**
   The backend pump calls it (5 sites): one engine, two harnesses —
   production pump and pipesim. That is good architecture, kept. The
   residual question is only whether the engine↔pump seam carries
   duplicate state bookkeeping; audit by reading the 5 call sites.
   Stake revised: ~−0.3K at most.
2. **`api/service.c`: RESOLVED — keeper.** Reading the vocabulary
   settled it: ClientSession, FrameHeader, EventFrame, Cancel,
   Disconnect, GetStats — this is residentd's control-plane wire
   protocol, a real domain distinct from request.c's user-request
   slots. Tested by its own gate. Future nicety only: a name that says
   protocol.
3. **`ring/transport/tcp.cu`: RESOLVED — deleted.** No selector, no
   test, and RDMA is universal on target hardware; memlink covers
   single-host, soft-RoCE (rxe) covers RDMA-less development. A
   fallback nothing can reach is not a fallback. −1,225 lines; the
   contract string went with it so the absence is loud.
4. **Analysis tools: RESOLVED — reclassified.** Ruling: studies and
   analyses are the tests category. The four analysis binaries and the
   K3 expert-subspace study live in tests/studies/ now; the Makefile
   follows; the denominator dropped 1,241 lines the honest way.
5. **node twins: RESOLVED, with a precise anatomy.** rank_daemon
   (3,088L/79 fns) is the ring-side bridge: transport to peers,
   connection to the local residentd, driver load, work-control pump.
   residentd (2,658L/55 fns) is the CUDA owner: node contexts, decode
   work packets, the client protocol, submit execution, stats. Shared
   POSTURE (daemon boilerplate ~450L combined: init, wake pipes,
   transport open, event wait, print-ready) — not shared WORK (~5.2K
   role-specific). An 80% parameterization does not exist to extract.
   The one real extraction: both ApplyArgument fns are if-strcmp
   chains (147L + 270L); a declarative option-table parser would make
   each a table, net ~−250L and LESS complexity. Queued in techdebt.
6. **`deployment/` (2.9K)**: release tooling. Earns its keep only if
   releases use it; audit at first release.

Rule reaffirmed: model drivers are never size cuts. The audits above
touch machinery only.

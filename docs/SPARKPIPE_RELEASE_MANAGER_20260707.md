# SparkPipe release manager

The production update path is now manifest driven. Spark2 owns one release directory containing `sparkpipe.json` and all common release files. Every Spark node runs a local `sparkpipe_release_manager agent` process for its role. The agent polls the manifest, verifies SHA-256 hashes, atomically installs changed files, stops the old role when required, and starts the exact command line from the manifest.

## Spark2 release directory

Example layout:

```text
/srv/sparkpipe/releases/current/
    sparkpipe.json
    bin/sparkpipe_glm52_http_gateway
    bin/sparkpipe_glm52_pp13_rank_daemon
    bin/sparkpipe_release_manager
    lib/libglm52_pp13_service_backend.so
    lib/libglm52_pp13_node_context_builder.so
    lib/libhidden_transport_tcp_cuda.so
    lib/model_driver.so
    packs/fp8/...
    packs/stage/...
    tokenizer/tokenizer.json
```

Publish a new release by writing files into that directory and then atomically replacing `sparkpipe.json`. The manifest generation and hashes are the source of truth.

## Generate a starter manifest

```sh
build/sparkpipe_release_manager example \
    --output /srv/sparkpipe/releases/current/sparkpipe.json
```

Replace every zero hash with the real SHA-256 of the corresponding file. The manifest supports these placeholders in commands, args, env, pid files, install roots, and state roots:

```text
{install_root}
{state_root}
{host}
{rank}
{rank_hex}
{rank_count}
{release_id}
{generation}
{role}
```

## Validate and inspect command lines

```sh
build/sparkpipe_release_manager validate \
    --manifest /srv/sparkpipe/releases/current/sparkpipe.json

build/sparkpipe_release_manager plan \
    --manifest /srv/sparkpipe/releases/current/sparkpipe.json \
    --host spark7 \
    --rank 7 \
    --role pp13_rank_daemon
```

## Serve the release directory from spark2

```sh
/home/spark2/sparkpipe_runtime/bin/sparkpipe_release_manager serve \
    --release-dir /srv/sparkpipe/releases/current \
    --bind 0.0.0.0 \
    --port 55420
```

## Rank agents

Run one agent per production role. Spark0 normally runs a gateway agent and a rank-daemon agent. Other ranks run only the rank-daemon agent.

Spark0 gateway:

```sh
/home/spark0/sparkpipe_runtime/bin/sparkpipe_release_manager agent \
    --release-url http://spark2:55420/ \
    --staging-dir /home/spark0/sparkpipe_state/release_staging \
    --install-dir /home/spark0/sparkpipe_runtime \
    --state-dir /home/spark0/sparkpipe_state \
    --host spark0 \
    --rank 0 \
    --role spark0_gateway \
    --poll-ms 1000
```

Rank daemon on `spark7`:

```sh
/home/spark7/sparkpipe_runtime/bin/sparkpipe_release_manager agent \
    --release-url http://spark2:55420/ \
    --staging-dir /home/spark7/sparkpipe_state/release_staging \
    --install-dir /home/spark7/sparkpipe_runtime \
    --state-dir /home/spark7/sparkpipe_state \
    --host spark7 \
    --rank 7 \
    --role pp13_rank_daemon \
    --poll-ms 1000
```

The agent does not use Python or shell orchestration. It is intended to run under systemd.

## CUDA pack residency

CUDA allocations do not survive a normal process restart because the GPU memory belongs to the process that allocated it. There are two practical production policies:

```text
1. Do not restart rank daemons when only gateway or non-rank files change.
2. Add a separate resident CUDA pack-cache process with CUDA IPC handles if packs must survive rank-daemon replacement.
```

The manifest already has the role/file flags needed for this policy:

```json
"allow_resident_pack_cache": true
"cuda_pack": true
"resident_reload_boundary": false
"restart_on_change": false
```

That lets a running rank keep already-loaded packs resident when the manifest update does not require an exact-generation rank restart. For a full driver or pack ABI change, the rank daemon still must restart and reload.

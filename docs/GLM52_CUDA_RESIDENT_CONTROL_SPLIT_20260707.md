# GLM-5.2 CUDA resident/control split

This release adds a real CUDA-resident data-plane process for PP13 ranks.

## Processes per rank

```text
sparkpipe_glm52_cuda_residentd
    long-lived CUDA owner
    loads FP8 packs, node-context builder, hidden transport, and the model_driver.so driver
    owns CUDA context, graphs, KV arenas, hidden transport sessions, and resident pack state
    accepts local Unix-socket work descriptors
    returns model-driver completions

sparkpipe_glm52_pp13_rank_daemon --cuda-resident-socket ...
    lightweight control process
    owns TCP work-control ring and final-event routing
    forwards PP13 work packets to the local resident daemon
    can be restarted while CUDA resident state remains warm
```

The old monolithic rank daemon path still exists if `--cuda-resident-socket` is not provided.

## Launch shape

Start residentd first on every rank:

```sh
bin/sparkpipe_glm52_cuda_residentd \
  --rank 7 \
  --socket /home/spark7/sparkpipe_state/cuda_resident_rank7.sock \
  --fp8-pack-root /home/spark7/sparkpipe_runtime/packs/fp8 \
  --stagepack-root /home/spark7/sparkpipe_runtime/packs/stage \
  --transport-so /home/spark7/sparkpipe_runtime/lib/libhidden_transport_tcp_cuda.so \
  --driver-so /home/spark7/sparkpipe_runtime/lib/model_driver.so \
  --program decode \
  --node-target cuda.sm121.glm52.resident_decode_stage.bf16 \
  --node-context-builder-so /home/spark7/sparkpipe_runtime/lib/libglm52_pp13_node_context_builder.so \
  --embedding-pack /home/spark7/sparkpipe_runtime/packs/embedding.sp \
  --max-active 1024 \
  --port-base 52100
```

Then start the control daemon:

```sh
bin/sparkpipe_glm52_pp13_rank_daemon \
  --rank 7 \
  --cuda-resident-socket /home/spark7/sparkpipe_state/cuda_resident_rank7.sock \
  --max-active 1024 \
  --port-base 52100 \
  --final-event-return-host spark0
```

## Update behavior

Control-only update:

```text
restart sparkpipe_glm52_pp13_rank_daemon only
CUDA residentd keeps packs, driver, graphs, KV, and transport resident
```

CUDA-level update:

```text
drain rank
restart sparkpipe_glm52_cuda_residentd
reload packs/driver/graphs/KV state as required
restart/reconnect control daemon
```

The example release manifest now contains two rank roles:

```text
pp13_cuda_residentd
pp13_rank_daemon
```

Run one release-agent instance for each role on every rank, or let systemd launch `pp13_cuda_residentd` as a stable service and use the release agent for `pp13_rank_daemon` while debugging control flow.

## IPC contract

The control daemon talks to residentd over a local Unix stream socket using fixed binary messages from:

```text
include/sparkpipe/spark_glm52_cuda_resident_ipc.h
```

Current messages:

```text
HELLO / HELLO_ACK
QUERY / STATS
SUBMIT_WORK / SUBMIT_RESULT
COMPLETION
SHUTDOWN
```

The hot path is currently Unix-socket binary messages. Once B1/B16/B128 are stable, the same ABI can be moved to a shared-memory ring plus eventfd without changing the PP13 work-control ring.

## Ownership boundary

`cuda_residentd` owns all CUDA-facing state. The control daemon should not load the node-context builder, model driver, hidden transport `.so`, CUDA packs, or CUDA graphs when `--cuda-resident-socket` is used.

## Operational notes

Driver artifact name: the driver `.so` is emitted by the driver compiler as
`model_driver.so` (SPARK_DRIVER_SHARED_OBJECT_NAME). Earlier manifests and this
doc referenced `glm5_2.fp8.so`, which is not a produced artifact; use
`lib/model_driver.so`.

`--program` value: the residentd, the old rank daemon, and the gateway all
default `--program` to `glm52.pp13.rank.production` in source. The
production-tested release used `--program decode`, which is the value shown
above. The program string is resolved against whatever the loaded driver `.so`
registers, so the correct value is deployment-specific; confirm against the
driver actually shipped in the release rather than assuming the source default.

spark0 resident init: the resident builder allocates fixed per-layer KV pools
sized by SPARK_GLM52_KV_POOL_TOKENS (4194304) with no runtime override, so every
rank reserves the same multi-GB pool per layer regardless of need. spark0 also
hosts the gateway/control plane, so it has the least free device memory and is
the only node that can fail `allocate_layer_buffers` partway through (observed at
layer 4, status IO_ERROR). The builder now logs requested and free/total device
bytes on any cudaMalloc failure (`pp13_builder_cuda_alloc_failed`); that line
confirms or rules out exhaustion. A runtime pool-size override is the mitigation
once the numbers confirm exhaustion, and is tracked separately.
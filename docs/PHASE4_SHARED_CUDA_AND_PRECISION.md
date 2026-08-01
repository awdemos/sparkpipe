# Phase 4: Shared CUDA Foundation and Precision Closure

## Scope

Phase 4 repairs host-verifiable CUDA contracts that are shared by Kimi K3,
GLM 5.2, Qwen 3.6 27B, DeepSeek V4 Flash, DeepSeek V4 Pro and MiMo 2.5. It
also makes each mandatory model's intended precision recipe explicit and
fail-closed.

This phase does not claim CUDA execution correctness or production readiness.
Exact CUDA 13 compilation for `sm_121a`, target-hardware numerical comparison,
Compute Sanitizer, transport validation and performance receipts remain
mandatory.

## Shared scale ABI

Quantized GEMM no longer treats every scale plane as `float *`. `LmScaleTensor`
now names the scale encoding and the complete addressing contract:

- no scale;
- FP32 scale;
- UE4M3 dynamic activation scale;
- UE8M0 microscale weight scale;
- group, row and K-group strides;
- rows and K elements represented by one scale;
- total element capacity.

The runtime validates these descriptors before launch. Dynamic activation
scales and checkpoint weight scales therefore cannot be silently aliased or
indexed as though they had the same representation.

## TMA ownership and descriptor lifetime

The shared GEMM path now passes `CUtensorMap` descriptors by value to the
kernel. It no longer passes pointers to temporary host-stack descriptors.
Producer ownership is CTA-wide, and the mbarrier expected-arrival count matches
the single elected producer.

The source contract also rejects architecture ambiguity. The compile-only gate
uses explicit `compute_121a` to `sm_121a` code generation and retains PTX,
objects, `ptxas` reports and optional `cuobjdump` reports.

## Race-free packed output

Sub-byte output packing assigns one thread an exclusive eight-code output
block. Adjacent threads no longer issue overlapping 32-bit read-modify-write
stores for 4-, 6-, 7- or 8-bit formats.

## Mandatory precision contracts

### Kimi K3

- routed-expert weights: MXFP4 E2M1;
- routed-expert activations: BF16;
- non-expert tensors: BF16;
- accumulators: FP32.

This is a deliberate SparkPipe deployment variant. It differs from the native
post-training recipe described by Moonshot, which uses MXFP4 expert weights
with MXFP8 expert activations.

### GLM 5.2

- routed-expert weights: FP8 E4M3;
- routed-expert activations: BF16;
- attention, dense/shared FFN, routers and other non-expert tensors: BF16;
- accumulators: FP32.

The GLM unity surface now exposes BF16 non-expert execution separately from the
FP8-weight expert path. Package and stage-pack contracts reject an FP8
non-expert recipe.

### Qwen 3.6 27B

- weights and activations: BF16;
- accumulators: FP32.

A source contract checks pack, binder and execution surfaces for accidental
quantized defaults.

### DeepSeek V4 Flash and Pro

- routed-expert weights: checkpoint FP4/MXFP4 representation;
- routed-expert activations: dynamic FP8 E4M3;
- non-expert linear weights: checkpoint FP8 E4M3 block 128x128;
- non-expert activations: dynamic FP8 E4M3;
- accumulators: FP32.

Flash and Pro have separate generated geometry and separate CUDA compile
surfaces. Pro still needs an independent complete execution module.

## DSV4 cache allocation

The DSV4 allocator remains class-exact. Sliding-attention history, compressed
history and compressor/indexer state are separate arenas. A layer receives only
the arena class required by its generated Flash or Pro schedule rather than the
largest reservation used by any layer.

## Build and audit closure

This phase also restores the executable neutral-core boundary auditor, repairs
parallel object-directory creation, removes the stale optional W8LUT clean
sub-make, updates the current source-existence gate and makes CUDA-only gates
report an explicit skip when `nvcc` is absent.

The compile-only installer now targets the official CUDA 13.3.1 redistributable
manifest, verifies every component checksum and performs an exact `sm_121a`
probe after installation.

## Network scope

The initial deployment remains one rail:

1. direct physical ring for deterministic debug;
2. one MikroTik 804 with one 100 Gbit/s port per Spark;
3. future second switch and second independent rail only after single-rail
   correctness and performance receipts.

The dual-rail configuration remains fail-closed.

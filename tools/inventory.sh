#!/bin/sh
# Every code file, with what it does. One line each.
#
# Exists because a breakdown by directory says how much and not what, and the
# question "would we lose anything by deleting this" needs the second.
cd "$(dirname "$0")/.." || exit 1
describe() {
	case "$1" in
	*/kernels/mma.cuh) echo "tensor core atoms, fragment layouts verified against CUTLASS" ;;
	*/kernels/tma.cuh) echo "async tile staging, mbarrier primitives" ;;
	*/kernels/tile.cuh) echo "the staged pipeline; two stages, lookahead one" ;;
	*/kernels/layout.cuh) echo "tile and swizzle geometry, host-computable, no CUDA" ;;
	*/kernels/dtype.cuh) echo "element conversions" ;;
	*/kernels/gemm.cuh) echo "ONE GEMM: grouped, dense is one group, format is a trait" ;;
	*/kernels/kv.cuh) echo "paged KV: opaque bytes, growing or recurrent" ;;
	*/kernels/norm.cuh) echo "rms norm, silu-mul, quantise, fused norm+quantise, moe finalize" ;;
	*/kernels/attn.cuh) echo "rope, yarn, latent attention, sparse scoring, hierarchical selection" ;;
	*/kernels/project.cuh) echo "low-rank projection, absorbed projection, fused QKV split" ;;
	*/kernels/topk.cuh) echo "top-k both shapes: bitonic small, radix large" ;;
	*/kernels/head.cuh) echo "sampling head: candidates, commit, softmax" ;;
	*/kernels/speculate.cuh) echo "speculative verify and accept, greedy and sampled" ;;
	*/kernels/linear_attn.cuh) echo "delta rule decode, causal conv; GDN and KDA" ;;
	*/kernels/graph.cuh) echo "CUDA graph capture keyed by shape" ;;
	*/kernels/tensor_map.cuh) echo "TMA descriptor geometry" ;;
	*/kernels/formats/*) echo "weight format trait: stored width, mma, fragment decode" ;;
	*/llms/*/config.h) echo "model shapes and constants, no code" ;;
	*/llms/*/unity.cu) echo "kernel instantiations and C entry points, one TU" ;;
	*/llms/*/layer.cuh) echo "the decode layer as a sequence of kernel launches" ;;
	*/llms/*/bind.cu) echo "node context to layer buffers, and the layer loop" ;;
	*/llms/*/api.h) echo "the model's C ABI" ;;
	*/stage/module.c) echo "stage interface: init, execute, admit, snapshot, complete" ;;
	*/stage/serving_adapter.cu) echo "serving frames to stage slices" ;;
	*/stage/dispatch.cu) echo "launch dispatch" ;;
	*/stage/runner.c) echo "production runner" ;;
	*/stage/draft_backend.cu) echo "DSpark drafter: a mini model for speculation" ;;
	api/request.c) echo "session and slot lifecycle, stop conditions, speculative policy" ;;
	api/backend.c) echo "per-rank serving: sockets, event loop, decode lanes" ;;
	api/serving_engine.c) echo "the engine: queues, events, stats" ;;
	api/service.c) echo "the frame protocol over the engine" ;;
	api/http_gateway.c) echo "HTTP surface" ;;
	api/compat_api.c) echo "compatibility shim" ;;
	api/text/tokenizer.*) echo "tokenizer" ;;
	scheduler/scheduler.c) echo "admission and batching, zero model constants" ;;
	scheduler/work_control.c) echo "packet building, batch buckets, prefill chunking" ;;
	scheduler/speculation.c) echo "DSpark dispatch policy" ;;
	scheduler/stage_plan.c) echo "which rank owns which layers" ;;
	scheduler/long_context.c) echo "long-context admission" ;;
	cache/cache.h) echo "THE CACHE: arena, content-addressed sharing, JIT reserve" ;;
	cache/kv_cache.c) echo "legacy arena, superseded by cache.h" ;;
	cache/prefix_cache.c) echo "legacy prefix index, superseded by cache.h" ;;
	cache/store/*) echo "KV block store and client" ;;
	ring/sideband.h) echo "cross-rank payloads: index share, hidden tap, prefix indices" ;;
	ring/transport/hidden_transport.*) echo "hidden state between ranks" ;;
	ring/transport/tp_collective.*) echo "tensor-parallel collectives" ;;
	ring/transport/memlink.*) echo "shared memory link" ;;
	ring/rdma_verbs.cu) echo "RDMA backend" ;;
	ring/tcp.cu) echo "TCP backend" ;;
	ring/rank_runtime.c) echo "per-rank runtime" ;;
	runtime/launch.h) echo "launch planning: tile height, shared bytes, grid" ;;
	runtime/gemm.cuh) echo "the four CUDA calls a GEMM needs" ;;
	runtime/workspace.h) echo "workspace pool layout" ;;
	runtime/tensor_map.h) echo "cuTensorMapEncodeTiled" ;;
	runtime/linear_plan.cu) echo "linear plan binding" ;;
	runtime/json.c) echo "JSON parser" ;;
	runtime/filesystem.c) echo "filesystem wrapper" ;;
	runtime/pack/*) echo "weight pack: description, library, compiler, stagepack, check" ;;
	modules/*decode_stage.cu) echo "LEGACY the 27k decode stage" ;;
	modules/*node_context_builder*) echo "LEGACY weight binding, 10k" ;;
	*) echo "" ;;
	esac
}
for f in $(git ls-files | grep -E '\.(c|cu|cuh|h)$' | grep -vE '^(tests|tools|docs)/')
do
	d=$(describe "$f")
	[ -z "$d" ] && continue
	printf "  %-52s %5s  %s\n" "$f" "$(wc -l < "$f")" "$d"
done

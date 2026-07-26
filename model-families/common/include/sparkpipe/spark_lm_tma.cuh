#pragma once

// Tensor-memory-accelerator staging and mbarrier pipeline synchronisation.
//
// spark_lm_async_copy.cuh issues cp.async, which is a per-thread instruction:
// every thread in the CTA computes an address and issues its own transfer, and
// the pipeline is tracked by commit-group depth. That works, but it burns issue
// slots proportional to tile bytes and it cannot express a bounded 2D box.
//
// TMA moves a whole tile with ONE instruction from ONE thread, addressed by
// tensor coordinates rather than by a linear pointer, and completion is tracked
// by an mbarrier transaction count rather than by group depth. That is the
// structure CUTLASS's SM120 collective uses - see
// third_party/flashinfer/3rdparty/cutlass/include/cutlass/gemm/collective/
// sm120_mma_array_tma_blockwise_scaling.hpp, which static_asserts its
// GmemTiledCopy to SM90_TMA_LOAD and builds its MainloopPipeline from
// PipelineTmaAsync<Stages>. Matching that structure is the point of this file.
//
// Every PTX form below assembles against the shipping target; see
// tests/test_ptx_capability_gate.py, which fails the build if one stops.
//
// There is no cp.async fallback here on purpose. A fallback would silently turn
// a one-instruction tile fetch into a per-thread address computation loop and
// nothing would report the change. spark_lm_async_copy.cuh remains available as
// an explicit, separately selected staging path.

#include <cuda_runtime.h>
#include <stdint.h>

// A TMA box is addressed in elements, but the transaction count an mbarrier
// expects is in bytes, and the two are only consistent if the caller derives
// one from the other. SparkLmTmaBoxBytes is the single place that conversion
// happens.
#define SPARK_LM_TMA_ALIGNMENT_BYTES 128u

static __device__ __forceinline__ uint32_t SparkLmTmaSharedAddress(const void *shared_pointer)
{
	return(static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void *>(shared_pointer))));
}

static __device__ __forceinline__ uint32_t SparkLmTmaBoxBytes(uint32_t rows, uint32_t columns, uint32_t element_bytes)
{
	return(rows * columns * element_bytes);
}

// One thread per CTA issues the TMA. elect.sync picks it from the leader warp
// without a ballot or a shared counter, and returns a predicate the caller
// branches on. Any thread of the warp may be elected; which one is irrelevant.
static __device__ __forceinline__ bool SparkLmTmaElectOne(void)
{
	uint32_t elected;
	asm volatile("{\n\t.reg .pred P;\n\t.reg .b32 L;\n\telect.sync L|P, 0xffffffff;\n\tselp.b32 %0, 1, 0, P;\n\t}\n"
		: "=r"(elected));
	return(elected != 0u);
}

static __device__ __forceinline__ void SparkLmMbarrierInit(uint64_t *barrier, uint32_t arrive_count)
{
	asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
		:: "r"(SparkLmTmaSharedAddress(barrier)), "r"(arrive_count));
}

static __device__ __forceinline__ void SparkLmMbarrierInvalidate(uint64_t *barrier)
{
	asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
		:: "r"(SparkLmTmaSharedAddress(barrier)));
}

// An mbarrier is initialised by one thread and read by all of them, and the
// initialising store is in the generic proxy while the TMA completion write is
// in the async proxy. Without this fence the two are not ordered and a consumer
// can observe an uninitialised barrier.
static __device__ __forceinline__ void SparkLmMbarrierInitFence(void)
{
	asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// Arrive and declare how many bytes this phase will receive. The barrier flips
// phase when the arrive count is met AND the byte count has landed, so the
// declared total must equal the sum of every box issued into this stage or the
// pipeline deadlocks. Callers derive it from SparkLmTmaBoxBytes rather than
// writing a literal.
static __device__ __forceinline__ void SparkLmMbarrierArriveExpect(uint64_t *barrier, uint32_t transaction_bytes)
{
	uint64_t state;
	asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
		: "=l"(state) : "r"(SparkLmTmaSharedAddress(barrier)), "r"(transaction_bytes));
}

static __device__ __forceinline__ void SparkLmMbarrierArrive(uint64_t *barrier)
{
	uint64_t state;
	asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
		: "=l"(state) : "r"(SparkLmTmaSharedAddress(barrier)));
}

// try_wait returns a predicate rather than blocking, so the spin lives in C++
// where it needs no PTX label. Inline-asm labels collide when a function is
// inlined more than once in a translation unit; returning the predicate avoids
// that class of failure entirely.
static __device__ __forceinline__ bool SparkLmMbarrierTryWait(uint64_t *barrier, uint32_t phase)
{
	uint32_t ready;
	asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\tselp.b32 %0, 1, 0, P;\n\t}\n"
		: "=r"(ready) : "r"(SparkLmTmaSharedAddress(barrier)), "r"(phase));
	return(ready != 0u);
}

static __device__ __forceinline__ void SparkLmMbarrierWait(uint64_t *barrier, uint32_t phase)
{
	while ( SparkLmMbarrierTryWait(barrier,phase) == false )
		;
}

// Load a 2D box. The tensor map is a CUtensorMap built on the host by
// cuTensorMapEncodeTiled and passed as a __grid_constant__ parameter; it encodes
// the global base, the element type, the box shape and the swizzle, so none of
// those appear here. Coordinates are in elements and are bounds-checked by the
// hardware, which zero-fills out-of-range elements - that is what makes a ragged
// group tail safe with no branch and no separate epilogue kernel.
static __device__ __forceinline__ void SparkLmTmaLoad2d(void *shared_destination, const void *tensor_map, uint64_t *barrier, int32_t coordinate_0, int32_t coordinate_1)
{
	asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
		:: "r"(SparkLmTmaSharedAddress(shared_destination)), "l"(tensor_map),
		   "r"(SparkLmTmaSharedAddress(barrier)), "r"(coordinate_0), "r"(coordinate_1)
		: "memory");
}

// Load a 3D box. Expert-major weights are exactly this: coordinate 2 selects the
// expert, so one descriptor covers all 256 of them and the grouped dispatch
// never rebuilds a tensor map per group.
static __device__ __forceinline__ void SparkLmTmaLoad3d(void *shared_destination, const void *tensor_map, uint64_t *barrier, int32_t coordinate_0, int32_t coordinate_1, int32_t coordinate_2)
{
	asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
		:: "r"(SparkLmTmaSharedAddress(shared_destination)), "l"(tensor_map),
		   "r"(SparkLmTmaSharedAddress(barrier)), "r"(coordinate_0), "r"(coordinate_1), "r"(coordinate_2)
		: "memory");
}

// Shared -> global for the epilogue. Completion is tracked by bulk group depth,
// not by an mbarrier, because nothing waits on the store except the next reuse
// of the staging buffer.
static __device__ __forceinline__ void SparkLmTmaStore2d(const void *tensor_map, const void *shared_source, int32_t coordinate_0, int32_t coordinate_1)
{
	asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
		:: "l"(tensor_map), "r"(SparkLmTmaSharedAddress(shared_source)), "r"(coordinate_0), "r"(coordinate_1)
		: "memory");
}

static __device__ __forceinline__ void SparkLmTmaStoreCommit(void)
{
	asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

// Retire all but KEEP outstanding store groups. .read is the weaker and cheaper
// form: it only guarantees the source shared memory is readable again, which is
// the actual requirement before overwriting a staging buffer.
template<uint32_t KEEP>
static __device__ __forceinline__ void SparkLmTmaStoreWait(void)
{
	asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(KEEP) : "memory");
}

// Writes into shared memory that a TMA store will read happen in the generic
// proxy; the store reads in the async proxy. This orders the two.
static __device__ __forceinline__ void SparkLmTmaStoreFence(void)
{
	asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

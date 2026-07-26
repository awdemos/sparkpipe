#pragma once

// Paged KV storage. Model-independent by construction.
//
// The cache does not know what a slot contains. It stores KV_SLOT_BYTES of
// opaque bytes per (sequence, position) and the driver decides what goes in
// them. That one decision makes every model in this tree use the same
// allocator, the same page table, the same block resolution and the same
// eviction:
//
//   MLA latent row      (512 + 64) elements x 2 bytes = 1152    grows
//   full KV head rows   heads x (key + value) x 2               grows
//   sliding-window rows same, but the pool is bounded by the window
//   recurrent state     fixed per sequence, one slot, never grows
//
// The last line is why this is worth doing. A linear-attention model carries
// fixed state instead of a growing cache, and the temptation is to give it its
// own subsystem. It does not need one: it is a pool with one slot per sequence
// and KV_GROWS false. The scheduler asks the same questions of both and gets
// answers in the same units.
//
// PAGING. Slots are grouped into pages of KV_PAGE_SLOTS so a sequence occupies
// a list of page indices rather than a contiguous range. That is what allows a
// prefix to be shared between sequences without copying it, and what allows a
// sequence to grow without moving. The page table is per-sequence and indexed
// by position / KV_PAGE_SLOTS.
//
// WHAT IS COMPILE TIME AND WHY. Slot size, page size and growth are template
// parameters, not fields. A runtime slot size puts a multiply in the address
// computation of every cache read, and cache reads are the whole cost of decode
// attention - the calibration measured three structurally different attention
// kernels producing identical time, which is the signature of a path bound only
// by the bytes all versions share. Nothing belongs in that address computation
// that a constant can answer.

// stdint only. A cache that stores opaque bytes has no business knowing about
// element formats, and depending on dtype.cuh would drag CUDA intrinsics into
// every host tool that wants to reason about capacity.
#include <stdint.h>

// A page table entry is a page index. UINT32_MAX means unmapped, which is
// distinct from page zero and is checked rather than assumed - an unmapped page
// read as page zero returns another sequence's keys and produces fluent wrong
// output.
#define LM_KV_PAGE_UNMAPPED 0xffffffffu

template<uint32_t SLOT_BYTES, uint32_t PAGE_SLOTS, bool GROWS>
struct LmKvGeometry
{
	static constexpr uint32_t kSlotBytes = SLOT_BYTES;
	static constexpr uint32_t kPageSlots = PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = SLOT_BYTES * PAGE_SLOTS;
	static constexpr bool kGrows = GROWS;

	// A page is the unit of sharing, so it must be a whole number of slots and
	// large enough that the page table is not itself a significant read. It must
	// also be a power of two: the position-to-page division is on the critical
	// path of every cache access and a shift is not a divide.
	static_assert(PAGE_SLOTS != 0u && (PAGE_SLOTS & (PAGE_SLOTS - 1u)) == 0u,
		"page size must be a power of two so position/page is a shift");
	static_assert(SLOT_BYTES % 16u == 0u,
		"slot must be 16-byte aligned so a cache read is a vector load");
	static_assert(GROWS || PAGE_SLOTS == 1u,
		"a non-growing pool holds one slot per sequence; pages have no meaning");

	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
	{
		return(position / PAGE_SLOTS);
	}

	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
	{
		return(position % PAGE_SLOTS);
	}

	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
	{
		return((tokens + PAGE_SLOTS - 1u) / PAGE_SLOTS);
	}

	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
	{
		return(pages * (uint64_t)kPageBytes);
	}
};

// A view is what a kernel receives. It carries no ownership and no allocation
// logic; those are host concerns and live in runtime/.
struct LmKvView
{
	uint8_t *pool;
	const uint32_t *page_table;      // [sequence][page] -> page index
	uint32_t page_table_stride;      // entries per sequence
	uint32_t sequence_count;
};

// Address of one slot. The only place cache addressing happens.
//
// Returns null for an unmapped page rather than an address into page zero. Every
// caller checks; the alternative is reading another sequence's keys and getting
// output that is fluent and wrong.
template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmKvSlot(const LmKvView &view, uint32_t sequence, uint32_t position)
{
	uint32_t page = view.page_table[(sequence * view.page_table_stride) + Geometry::PageOf(position)];
	if ( page == LM_KV_PAGE_UNMAPPED )
		return(0);
	return(view.pool + ((uint64_t)page * Geometry::kPageBytes)
		+ ((uint64_t)Geometry::SlotInPage(position) * Geometry::kSlotBytes));
}

template<class Geometry>
static __device__ __forceinline__ uint8_t *LmKvSlotMutable(const LmKvView &view, uint32_t sequence, uint32_t position)
{
	return((uint8_t *)LmKvSlot<Geometry>(view,sequence,position));
}

// Slots a sequence occupies, for a scheduler that needs to know before it reads.
template<class Geometry>
static __host__ __device__ __forceinline__ uint64_t LmKvBytesForSequence(uint32_t length)
{
	return((uint64_t)Geometry::PagesForTokens(length) * (uint64_t)Geometry::kPageBytes);
}

// -- concrete geometries -----------------------------------------------------
//
// Named here rather than in each model header so two models with the same shape
// provably share one instantiation. A model header supplies the numbers; this
// file supplies the type.
//
// THE CACHE FORMAT IS INDEPENDENT OF THE WEIGHT FORMAT, and both come from
// kernels/formats/. NVFP4 weights with an FP8 cache is a normal combination:
// weights are read once per tile and shared across every row in it, while the
// cache is read once per (sequence, slot) and shared with nothing, so the two
// sit at different points on the precision-versus-bytes curve and should be
// chosen separately.
//
// The cache itself still stores opaque bytes and knows nothing about the format.
// It appears here only to compute the slot size; the attention kernel that reads
// a slot is what decodes it, and it takes the same trait.
//
// BITS is passed rather than the trait so this header keeps its stdint-only
// dependency - a host tool sizing a pool should not need a CUDA toolchain. A
// model writes LmKvLatent<LmFp8::kBits, ...> and the two stay tied.

// Latent attention: one shared row per slot regardless of head count. The whole
// reason a 64-head model is tractable at decode.
template<uint32_t ELEMENT_BITS, uint32_t LATENT_ELEMENTS, uint32_t ROPE_ELEMENTS, uint32_t PAGE_SLOTS>
using LmKvLatent = LmKvGeometry<(((LATENT_ELEMENTS + ROPE_ELEMENTS) * ELEMENT_BITS) / 8u), PAGE_SLOTS, true>;

// Per-head key and value rows, for models without latent compression.
template<uint32_t ELEMENT_BITS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t PAGE_SLOTS>
using LmKvHeads = LmKvGeometry<((KV_HEADS * HEAD_DIM * 2u * ELEMENT_BITS) / 8u), PAGE_SLOTS, true>;

// Recurrent state: one slot per sequence, never grows, no paging.
template<uint32_t STATE_BYTES>
using LmKvState = LmKvGeometry<STATE_BYTES, 1u, false>;

// -- writing ------------------------------------------------------------------
//
// A cache that is only ever read is a cache full of whatever the allocator left
// behind. Attention over it produces fluent output from garbage keys, which
// looks like an attention bug and is not one.
//
// This is here rather than in whatever computes the row because the slot
// addressing is here, and an addressing scheme with two implementations is an
// addressing scheme with two chances to be wrong.
#ifdef __CUDACC__
template<class Geometry, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvStoreKernel(LmKvView view, const uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count, uint32_t elements)
{
	uint32_t row = blockIdx.x,index;
	uint8_t *slot;
	if ( row >= row_count )
		return;
	slot = LmKvSlotMutable<Geometry>(view,sequence_of_row[row],position_of_row[row]);
	// An unmapped page here is a page the scheduler failed to allocate, which is
	// a hard error rather than something to skip: skipping loses the token's key
	// silently and every later step attends over a hole.
	if ( slot == 0 )
		return;
	for (index = threadIdx.x; index < elements; index += THREADS)
		((uint16_t *)slot)[index] = rows_bf16[((uint64_t)row * elements) + index];
}
#endif

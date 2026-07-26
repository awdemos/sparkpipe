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

// Latent attention: one shared row per slot regardless of head count. The whole
// reason a 64-head model is tractable at decode.
template<uint32_t LATENT_ELEMENTS, uint32_t ROPE_ELEMENTS, uint32_t PAGE_SLOTS>
using LmKvLatent = LmKvGeometry<((LATENT_ELEMENTS + ROPE_ELEMENTS) * 2u), PAGE_SLOTS, true>;

// Per-head key and value rows, for models without latent compression.
template<uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t PAGE_SLOTS>
using LmKvHeads = LmKvGeometry<(KV_HEADS * HEAD_DIM * 2u * 2u), PAGE_SLOTS, true>;

// Recurrent state: one slot per sequence, never grows, no paging.
template<uint32_t STATE_BYTES>
using LmKvState = LmKvGeometry<STATE_BYTES, 1u, false>;

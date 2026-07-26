// KV geometry: one allocator, three models whose slots mean different things.
// Compile-only - every check below is a static_assert, so a wrong geometry is a
// build failure on the sparkdev's laptop rather than a launch failure on the ring.
#include <stdint.h>
#define __host__
#define __device__
#include "kernels/kv.cuh"
#include "llms/glm5_2/config.h"

using Glm52Kv    = LmKvLatent<GLM52_LATENT, GLM52_ROPE_DIM, GLM52_KV_PAGE_SLOTS>;
using Mimo25Full = LmKvHeads<4u, 128u, 64u>;
using Qwen36Gdn  = LmKvState<131072u>;

static_assert(Glm52Kv::kSlotBytes == 1152u, "MLA latent slot: (512+64) x bf16");
static_assert(Glm52Kv::kSlotBytes == GLM52_KV_SLOT_BYTES, "config.h agrees with kv.cuh");
static_assert(Glm52Kv::kPageBytes == 73728u, "64-slot page is 72 KB");
static_assert(Glm52Kv::PageOf(129u) == 2u, "position 129 is in page 2");
static_assert(Glm52Kv::SlotInPage(129u) == 1u, "position 129 is slot 1 of it");
static_assert(Glm52Kv::PagesForTokens(1048576u) == 16384u, "1M context is 16384 pages");
static_assert(Mimo25Full::kSlotBytes == 2048u, "4 KV heads x 128 dim x (k+v) x bf16");
static_assert(Qwen36Gdn::kGrows == false, "recurrent state does not grow");
static_assert(Qwen36Gdn::kPageSlots == 1u, "state is one slot per sequence");

// The same address helper serves a paged cache and a non-paged state pool.
void probe(void)
{
	LmKvView view;
	view.pool = 0;
	view.page_table = 0;
	view.page_table_stride = 0;
	view.sequence_count = 0;
	(void)LmKvSlot<Glm52Kv>(view, 0u, 0u);
	(void)LmKvSlot<Qwen36Gdn>(view, 0u, 0u);
	(void)LmKvBytesForSequence<Glm52Kv>(1024u);
}

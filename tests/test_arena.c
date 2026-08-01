// The arena's whole contract: one allocation at init and none after, every
// slot reachable exactly once, exhaustion a loud NULL rather than a quiet
// malloc fallback, double release and foreign pointers refused, slots
// 16-byte aligned, and the acquire generation advancing so a caller can
// detect slot reuse across an ABA window.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "runtime/arena.h"

static int32_t SparkTestArenaCycles(void)
{
	SparkArena arena;
	SparkArenaClassDescriptor classes[2];
	void *first[3];
	void *second;
	void *third;
	uint32_t index;

	classes[0u].slot_bytes = 100u;
	classes[0u].slot_count = 3u;
	classes[1u].slot_bytes = 4096u;
	classes[1u].slot_count = 2u;
	assert(SparkArenaInitialize(&arena,classes,2u) == 0);
	assert(SparkArenaReservedBytes(&arena) ==
		3ull * 112ull + 2ull * 4096ull +
			(3ull + 2ull) * (uint64_t)sizeof(uint32_t));

	/* Small requests route to the smallest fitting class only. */
	for (index = 0u; index < 3u; ++index)
	{
		first[index] = SparkArenaAcquire(&arena,(uint32_t)(index + 1u));
		assert(first[index] != 0);
		assert(((uintptr_t)first[index] % SPARK_ARENA_ALIGNMENT) == 0u);
		memset(first[index],(int)(index + 1u),(size_t)(index + 1u));
	}
	/* Exhaustion: the class is dry and the answer is NULL - there is no
	   malloc path inside acquire to fall back to, by construction. */
	assert(SparkArenaAcquire(&arena,1u) == 0);
	/* No silent promotion into the larger class either. */
	assert(SparkArenaClassGeneration(&arena,0u) == 3u);

	/* A large request lands in the large class, aligned, disjoint. */
	second = SparkArenaAcquire(&arena,4096u);
	third = SparkArenaAcquire(&arena,2000u);
	assert(second != 0 && third != 0);
	assert(((uintptr_t)second % SPARK_ARENA_ALIGNMENT) == 0u);
	assert((uint8_t *)third >= (uint8_t *)second + 4096 ||
		(uint8_t *)second >= (uint8_t *)third + 4096);
	assert(SparkArenaAcquire(&arena,4096u) == 0);
	/* Too large for every class: NULL, not malloc. */
	assert(SparkArenaAcquire(&arena,8192u) == 0);

	/* Release and reacquire: exact slot reuse, content readable until
	   then, generation advanced. */
	assert(SparkArenaRelease(&arena,first[1u]) == 0);
	assert(SparkArenaAcquire(&arena,64u) == first[1u]);
	assert(SparkArenaClassGeneration(&arena,0u) == 4u);
	assert(*(uint8_t *)first[0u] == 1u);

	/* Double release and foreign pointers are refused. */
	assert(SparkArenaRelease(&arena,second) == 0);
	assert(SparkArenaRelease(&arena,second) == -30908);
	assert(SparkArenaRelease(&arena,(void *)(uintptr_t)0x10000u) ==
		-30909);
	/* Interior pointer of a live slot is not a slot start. */
	assert(SparkArenaRelease(&arena,(uint8_t *)third + 16u) == -30907);

	SparkArenaDestroy(&arena);
	assert(arena.backing == 0);
	/* Destroy on a zeroed arena is a no-op (failed-init path). */
	SparkArenaDestroy(&arena);
	return 0;
}

static int32_t SparkTestArenaRejectsBadGeometry(void)
{
	SparkArena arena;
	SparkArenaClassDescriptor classes[2];

	memset(&arena,0,sizeof(arena));
	/* Descending classes would route small allocations into big slots. */
	classes[0u].slot_bytes = 4096u;
	classes[0u].slot_count = 1u;
	classes[1u].slot_bytes = 64u;
	classes[1u].slot_count = 1u;
	assert(SparkArenaInitialize(&arena,classes,2u) == -30903);
	assert(arena.backing == 0);
	/* Empty class. */
	classes[0u].slot_count = 0u;
	assert(SparkArenaInitialize(&arena,classes,1u) == -30902);
	/* Overflowing geometry: 2^32-byte slots round up to 2^32, and
	   (2^32 + 4) * (2^32 - 1) exceeds UINT64_MAX, so the accumulator
	   check must fire before any malloc is attempted. */
	classes[0u].slot_bytes = 0xffffffffu;
	classes[0u].slot_count = 0xffffffffu;
	assert(SparkArenaInitialize(&arena,classes,1u) == -30904);
	assert(arena.backing == 0);
	SparkArenaDestroy(&arena);
	return 0;
}

int main(void)
{
	SparkTestArenaCycles();
	SparkTestArenaRejectsBadGeometry();
	printf("arena: cycles, alignment, loud exhaustion, reuse, geometry\n");
	return 0;
}

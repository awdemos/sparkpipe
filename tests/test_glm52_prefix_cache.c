#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_prefix_cache.h"

static void SparkTestInitializePrefixCache(
    SparkPrefixCache *cache,
    SparkPrefixCacheEntry *entries,
    SparkPrefixCacheSequenceBinding *bindings,
    uint32_t entry_count,
    uint32_t binding_count,
    uint32_t block_token_count)
{
    SparkPrefixCacheConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.block_token_count = block_token_count;
    configuration.entry_count = entry_count;
    configuration.physical_block_count = entry_count;
    configuration.sequence_binding_count = binding_count;
    configuration.entries = entries;
    configuration.sequence_bindings = bindings;
    assert(SparkPrefixCacheInitialize(cache, &configuration) ==
        SPARK_STATUS_OK);
}

static void SparkTestPrefixCacheMatchesCommittedBlocks(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[8u];
    SparkPrefixCacheSequenceBinding bindings[16u];
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[16u];
    uint32_t token_index;

    for (token_index = 0u; token_index < 16u; ++token_index)
    {
        tokens[token_index] = 1000u + token_index;
    }

    SparkTestInitializePrefixCache(&cache, entries, bindings, 8u, 16u, 4u);
    assert(SparkPrefixCacheLookupPrompt(
        &cache,
        7u,
        tokens,
        16u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 0u);
    assert(cache.miss_count == 1u);

    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        7u,
        tokens,
        16u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 16u);
    assert(lookup.matched_block_count == 4u);
    assert(cache.inserted_block_count == 4u);
    assert(cache.acquired_block_count == 4u);

    assert(SparkPrefixCacheLookupPrompt(
        &cache,
        9u,
        tokens,
        16u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 12u);
    assert(lookup.matched_block_count == 3u);
    assert(lookup.next_token_index == 12u);
    assert(cache.hit_count == 1u);
    assert(cache.acquired_block_count == 7u);
}

static void SparkTestPrefixCacheStopsAtChangedBlock(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[8u];
    SparkPrefixCacheSequenceBinding bindings[16u];
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[16u];
    uint32_t changed_tokens[16u];
    uint32_t token_index;

    for (token_index = 0u; token_index < 16u; ++token_index)
    {
        tokens[token_index] = 2000u + token_index;
        changed_tokens[token_index] = tokens[token_index];
    }
    changed_tokens[6u] = 9000u;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 8u, 16u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        12u,
        tokens,
        16u,
        0) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheLookupPrompt(
        &cache,
        13u,
        changed_tokens,
        16u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 4u);
    assert(lookup.matched_block_count == 1u);
}

static void SparkTestPrefixCacheTracksSequenceOwnership(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[4u];
    SparkPrefixCacheSequenceBinding bindings[8u];
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[8u] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};

    SparkTestInitializePrefixCache(&cache, entries, bindings, 4u, 8u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        1u,
        tokens,
        8u,
        0) == SPARK_STATUS_OK);
    assert(entries[0u].reference_count == 1u);
    assert(entries[1u].reference_count == 1u);

    assert(SparkPrefixCacheLookupPrompt(
        &cache,
        2u,
        tokens,
        8u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 4u);
    assert(entries[0u].reference_count == 2u);
    assert(entries[1u].reference_count == 1u);

    assert(SparkPrefixCacheReleaseSequence(&cache, 2u) == SPARK_STATUS_OK);
    assert(entries[0u].reference_count == 1u);
    assert(entries[1u].reference_count == 1u);
    assert(SparkPrefixCacheReleaseSequence(&cache, 1u) == SPARK_STATUS_OK);
    assert(entries[0u].reference_count == 0u);
    assert(entries[1u].reference_count == 0u);
    assert(cache.released_block_count == 3u);
}

static void SparkTestPrefixCacheEvictsReleasedBlocks(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[2u];
    SparkPrefixCacheSequenceBinding bindings[8u];
    uint32_t first_tokens[8u] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    uint32_t second_tokens[8u] = {9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};
    uint32_t third_tokens[8u] = {17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u};

    SparkTestInitializePrefixCache(&cache, entries, bindings, 2u, 8u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        1u,
        first_tokens,
        8u,
        0) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        2u,
        second_tokens,
        8u,
        0) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkPrefixCacheReleaseSequence(&cache, 1u) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        2u,
        second_tokens,
        8u,
        0) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheReleaseSequence(&cache, 2u) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        3u,
        third_tokens,
        8u,
        0) == SPARK_STATUS_OK);
    assert(cache.evicted_block_count != 0u);
}

static void SparkTestPrefixCacheRejectsBindingExhaustionWithoutLeakingRefs(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[4u];
    SparkPrefixCacheSequenceBinding bindings[2u];
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[16u];
    uint32_t token_index;

    for (token_index = 0u; token_index < 16u; ++token_index)
    {
        tokens[token_index] = 100u + token_index;
    }

    SparkTestInitializePrefixCache(&cache, entries, bindings, 4u, 2u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        1u,
        tokens,
        16u,
        0) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(entries[0u].reference_count == 0u);
    assert(entries[1u].reference_count == 0u);
    assert(SparkPrefixCacheReleaseSequence(&cache, 1u) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        2u,
        tokens,
        8u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 8u);
}


static void SparkTestPrefixCacheReservationOwnsPhysicalBlocksUntilCommitOrCancel(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[8u];
    SparkPrefixCacheSequenceBinding bindings[16u];
    SparkPrefixCacheReservation reservation;
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[12u];
    uint32_t physical_blocks[4u];
    uint32_t physical_block_count;
    uint32_t token_index;

    for (token_index = 0u; token_index < 12u; ++token_index)
    {
        tokens[token_index] = 3000u + token_index;
    }

    SparkTestInitializePrefixCache(&cache, entries, bindings, 8u, 16u, 4u);
    memset(&reservation, 0, sizeof(reservation));
    reservation.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    reservation.descriptor_bytes =
        SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    reservation.physical_block_indices = physical_blocks;
    reservation.physical_block_capacity = 4u;

    assert(SparkPrefixCacheReservePrompt(
        &cache,
        101u,
        tokens,
        12u,
        &reservation) == SPARK_STATUS_OK);
    assert(reservation.reserved_token_count == 12u);
    assert(reservation.physical_block_count == 3u);
    assert(reservation.pending_physical_block_count == 3u);
    assert(physical_blocks[0u] != SPARK_PREFIX_CACHE_NO_PHYSICAL_BLOCK);
    assert(physical_blocks[1u] != SPARK_PREFIX_CACHE_NO_PHYSICAL_BLOCK);
    assert(physical_blocks[2u] != SPARK_PREFIX_CACHE_NO_PHYSICAL_BLOCK);

    assert(SparkPrefixCacheBuildPhysicalBlockTable(
        &cache,
        101u,
        12u,
        physical_blocks,
        4u,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 3u);

    assert(SparkPrefixCacheProbePrompt(
        &cache,
        102u,
        tokens,
        12u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 0u);

    assert(SparkPrefixCacheCancelReservation(
        &cache,
        101u,
        reservation.reservation_epoch) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheBuildPhysicalBlockTable(
        &cache,
        101u,
        12u,
        physical_blocks,
        4u,
        &physical_block_count) == SPARK_STATUS_NOT_FOUND);

    memset(&reservation, 0, sizeof(reservation));
    reservation.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    reservation.descriptor_bytes =
        SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    reservation.physical_block_indices = physical_blocks;
    reservation.physical_block_capacity = 4u;
    assert(SparkPrefixCacheReservePrompt(
        &cache,
        103u,
        tokens,
        12u,
        &reservation) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheCommitReservation(
        &cache,
        103u,
        reservation.reservation_epoch) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheProbePrompt(
        &cache,
        104u,
        tokens,
        12u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 8u);
}


static void SparkTestPrefixCacheLookaheadProtectionSkipsProtectedVictim(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[2u];
    SparkPrefixCacheSequenceBinding bindings[8u];
    SparkPrefixCacheLookup lookup;
    uint32_t protected_prompt[8u] = {10u, 11u, 12u, 13u, 90u, 91u, 92u, 93u};
    uint32_t competing_prompt[4u] = {20u, 21u, 22u, 23u};
    uint32_t incoming_prompt[4u] = {30u, 31u, 32u, 33u};
    uint32_t protected_token_count;
    uint32_t protected_block_count;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 2u, 8u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        101u,
        protected_prompt,
        4u,
        0) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheReleaseSequence(&cache, 101u) ==
        SPARK_STATUS_OK);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        202u,
        competing_prompt,
        4u,
        0) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheReleaseSequence(&cache, 202u) ==
        SPARK_STATUS_OK);

    assert(SparkPrefixCacheResetLookaheadProtection(&cache) ==
        SPARK_STATUS_OK);
    assert(SparkPrefixCacheProtectPromptLookahead(
        &cache,
        protected_prompt,
        8u,
        500u,
        &protected_token_count,
        &protected_block_count) == SPARK_STATUS_OK);
    assert(protected_token_count == 4u);
    assert(protected_block_count == 1u);

    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        303u,
        incoming_prompt,
        4u,
        0) == SPARK_STATUS_OK);
    assert(cache.evicted_block_count == 1u);

    assert(SparkPrefixCacheProbePrompt(
        &cache,
        404u,
        protected_prompt,
        8u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 4u);

    assert(SparkPrefixCacheProbePrompt(
        &cache,
        505u,
        competing_prompt,
        8u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 0u);
}

static void SparkTestPrefixCacheChainedBlockHashMatchesFullHash(void)
{
    SparkPrefixCachePromptHash full_hash;
    SparkPrefixCachePromptHash block_hash;
    uint32_t token_ids[80u];
    uint32_t token_index;
    uint32_t boundary_token_count;
    uint64_t chained_hash;

    for (token_index = 0u; token_index < 80u; ++token_index)
    {
        token_ids[token_index] = 100000u + (token_index * 37u);
    }
    chained_hash = SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH;
    for (boundary_token_count = 16u;
         boundary_token_count <= 80u;
         boundary_token_count += 16u)
    {
        assert(SparkPrefixCacheHashPromptTokens(
            16u,
            SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH,
            token_ids,
            boundary_token_count,
            &full_hash) == SPARK_STATUS_OK);
        assert(SparkPrefixCacheHashPromptTokens(
            16u,
            chained_hash,
            &token_ids[boundary_token_count - 16u],
            16u,
            &block_hash) == SPARK_STATUS_OK);
        chained_hash = block_hash.prompt_hash;
        assert(chained_hash == full_hash.prompt_hash);
    }
}

static void SparkTestPrefixCacheExtendsLiveSequenceCapacityIdempotently(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[8u];
    SparkPrefixCacheSequenceBinding bindings[16u];
    uint32_t prompt[3u] = {41u, 42u, 43u};
    uint32_t first_table[4u];
    uint32_t second_table[4u];
    uint32_t first_block_count;
    uint32_t second_block_count;
    uint64_t inserted_block_count;
    uint32_t block_index;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 8u, 16u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache,
        707u,
        prompt,
        3u,
        0) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheBuildPhysicalBlockTable(
        &cache,
        707u,
        3u,
        first_table,
        4u,
        &first_block_count) == SPARK_STATUS_OK);
    assert(first_block_count == 1u);

    assert(SparkPrefixCacheEnsureSequenceTokenCapacity(
        &cache,
        707u,
        10u) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheBuildPhysicalBlockTable(
        &cache,
        707u,
        10u,
        first_table,
        4u,
        &first_block_count) == SPARK_STATUS_OK);
    assert(first_block_count == 3u);
    for (block_index = 0u; block_index < first_block_count; ++block_index)
    {
        SparkPrefixCacheEntry *entry;

        entry = &entries[first_table[block_index]];
        assert((entry->flags &
            SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) != 0u);
        assert((entry->flags &
            SPARK_PREFIX_CACHE_ENTRY_FLAG_REUSABLE) == 0u);
        assert(entry->token_count == 4u);
    }

    inserted_block_count = cache.inserted_block_count;
    assert(SparkPrefixCacheEnsureSequenceTokenCapacity(
        &cache,
        707u,
        10u) == SPARK_STATUS_OK);
    assert(cache.inserted_block_count == inserted_block_count);
    assert(SparkPrefixCacheBuildPhysicalBlockTable(
        &cache,
        707u,
        10u,
        second_table,
        4u,
        &second_block_count) == SPARK_STATUS_OK);
    assert(second_block_count == first_block_count);
    assert(memcmp(first_table, second_table,
        first_block_count * sizeof(first_table[0u])) == 0);
}

static void SparkTestPrefixCacheSequenceReservationDoesNotReuseContent(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[8u];
    SparkPrefixCacheSequenceBinding bindings[16u];
    SparkPrefixCacheReservation reservation;
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[12u];
    uint32_t first_blocks[3u];
    uint32_t second_blocks[3u];
    uint32_t token_index;

    for (token_index = 0u; token_index < 12u; ++token_index)
        tokens[token_index] = 9000u + token_index;
    SparkTestInitializePrefixCache(&cache,entries,bindings,8u,16u,4u);
    memset(&reservation,0,sizeof(reservation));
    reservation.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    reservation.descriptor_bytes =
        SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    reservation.physical_block_indices = first_blocks;
    reservation.physical_block_capacity = 3u;
    assert(SparkPrefixCacheReserveSequencePrompt(
        &cache,801u,tokens,8u,&reservation) == SPARK_STATUS_OK);
    assert(reservation.cached_physical_block_count == 0u);
    assert(SparkPrefixCacheCommitReservation(
        &cache,801u,reservation.reservation_epoch) == SPARK_STATUS_OK);
    memset(&reservation,0,sizeof(reservation));
    reservation.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
    reservation.descriptor_bytes =
        SPARK_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
    reservation.physical_block_indices = second_blocks;
    reservation.physical_block_capacity = 3u;
    assert(SparkPrefixCacheReserveSequencePrompt(
        &cache,802u,tokens,8u,&reservation) == SPARK_STATUS_OK);
    assert(reservation.cached_physical_block_count == 0u);
    assert(reservation.reusable_token_count == 0u);
    assert(first_blocks[0u] != second_blocks[0u]);
    assert(first_blocks[1u] != second_blocks[1u]);
    assert(SparkPrefixCacheCommitReservation(
        &cache,802u,reservation.reservation_epoch) == SPARK_STATUS_OK);
    assert(SparkPrefixCacheProbePrompt(
        &cache,803u,tokens,8u,&lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 0u);
    for (token_index = 0u; token_index < 2u; ++token_index)
        assert((entries[first_blocks[token_index]].flags &
            SPARK_PREFIX_CACHE_ENTRY_FLAG_LIVE_ONLY) != 0u);
}

// The three probes share one chain walk. Pin what that walk must produce:
// the same match length from every probe, a recency tick advanced once per
// matched block, and a capacity refusal that does not consume the chain.
static void SparkTestPrefixCacheProbesShareOneWalk(void)
{
    SparkPrefixCache cache;
    SparkPrefixCacheEntry entries[8u];
    SparkPrefixCacheSequenceBinding bindings[16u];
    SparkPrefixCacheLookup lookup;
    uint32_t tokens[16u];
    uint32_t physical_block_indices[8u];
    uint32_t token_index;
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    uint64_t tick_before;

    for (token_index = 0u; token_index < 16u; ++token_index)
    {
        tokens[token_index] = 4000u + token_index;
    }
    SparkTestInitializePrefixCache(&cache, entries, bindings, 8u, 16u, 4u);
    assert(SparkPrefixCacheCommitPrompt(
        &cache, 11u, tokens, 16u, &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_block_count == 4u);

    // The final block is withheld from reuse - a prompt must keep at least one
    // token to generate from - so 16 tokens at block size 4 yield three.
    tick_before = cache.tick;
    assert(SparkPrefixCacheProbePhysicalBlockTable(
        &cache, tokens, 16u, physical_block_indices, 8u,
        &matched_token_count, &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 3u);
    assert(matched_token_count == 12u);
    assert(cache.tick == tick_before + 3u);

    // A second probe over the same prompt must agree on the match length.
    assert(SparkPrefixCacheProbeReusablePrefixResidency(
        &cache, tokens, 16u, &matched_token_count,
        &resident_block_count, &nonresident_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == 12u);
    assert(resident_block_count + nonresident_block_count == 3u);

    // A short output refuses rather than truncating, and the refusal must not
    // leave the chain half-walked for the next caller.
    assert(SparkPrefixCacheProbePhysicalBlockTable(
        &cache, tokens, 16u, physical_block_indices, 2u,
        &matched_token_count, &physical_block_count) ==
            SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkPrefixCacheProbePhysicalBlockTable(
        &cache, tokens, 16u, physical_block_indices, 8u,
        &matched_token_count, &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 3u);
    assert(matched_token_count == 12u);

    // A prompt that diverges after one block matches exactly one block.
    tokens[4u] = 9999u;
    assert(SparkPrefixCacheProbePhysicalBlockTable(
        &cache, tokens, 16u, physical_block_indices, 8u,
        &matched_token_count, &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 1u);
    assert(matched_token_count == 4u);
}

int main(void)
{
    SparkTestPrefixCacheMatchesCommittedBlocks();
    SparkTestPrefixCacheProbesShareOneWalk();
    SparkTestPrefixCacheStopsAtChangedBlock();
    SparkTestPrefixCacheTracksSequenceOwnership();
    SparkTestPrefixCacheEvictsReleasedBlocks();
    SparkTestPrefixCacheRejectsBindingExhaustionWithoutLeakingRefs();
    SparkTestPrefixCacheReservationOwnsPhysicalBlocksUntilCommitOrCancel();
    SparkTestPrefixCacheLookaheadProtectionSkipsProtectedVictim();
    SparkTestPrefixCacheChainedBlockHashMatchesFullHash();
    SparkTestPrefixCacheExtendsLiveSequenceCapacityIdempotently();
    SparkTestPrefixCacheSequenceReservationDoesNotReuseContent();
    return 0;
}

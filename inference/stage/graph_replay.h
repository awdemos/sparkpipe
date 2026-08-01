#ifndef SPARKPIPE_INFERENCE_STAGE_GRAPH_REPLAY_H
#define SPARKPIPE_INFERENCE_STAGE_GRAPH_REPLAY_H

// Stage-side CUDA graph capture and replay for the resident decode step.
//
// A K3 decode token costs about 3,300 kernel launches across its layers, each
// a few microseconds of host-side driver work. At B1 the GPU finishes every
// kernel before the host can name the next one, so the token rate IS the host
// launch rate. A captured graph turns the step into one submission: the driver
// walks a dependency tree it validated at capture time and the host cost stops
// scaling with layer count.
//
// This file is the STAGE side of that seam. It owns three things and nothing
// else: what makes two steps interchangeable (the key), where recordings live
// (the fixed slots in SparkResidentDecodeStageCudaPipelineSlotState - no heap
// in steady state), and the capture/replay/fallback decision. The launches
// themselves belong to the driver layers; the CUDA calls belong to the
// dispatch translation unit. Both enter through the two tables below, which is
// what makes every non-CUDA branch of this logic host-testable.
//
// THE REPLAY CONTRACT - WHAT MAY VARY BETWEEN REPLAYS OF ONE GRAPH.
//
//   MAY vary, because a graph records pointers and re-reads their contents on
//   every replay: the CONTENTS of any device buffer captured by address -
//   token ids, positions, slot mappings, context lengths, KV block table
//   entries, hidden states, router outputs, MTP draft budgets, hidden
//   transport payloads. These are the device-side scalars; a decode step
//   exists to change them.
//
//   MUST NOT vary, because the value was folded into the recording: every
//   host-derived input to the launch sequence. Anything the driver layers read
//   on the host to pick a grid, a branch, or a kernel is baked in, and
//   replaying it against a different value reads and writes whatever the old
//   value meant. Every host-derived input the stage knows about is folded
//   into the key below: active sequence count, final-token stage, layer
//   count, frame flags (which encode transport/tap/budget presence),
//   lane geometry, the slice plan identity, and the KV block table's device
//   pointer. A driver layer that adds a host-derived input MUST extend the
//   signature - or it is not capturable, and must say so by leaving
//   enable_cuda_graph_replay clear.
//
// POINTER STABILITY. Every buffer the step touches must live at the same
// address on every replay. The stage guarantees this for everything it owns:
// pipeline slot buffers, plans, and caches are bound once at context build
// and never move. The one per-frame pointer, the KV block table view, is
// keyed rather than trusted: the table's DEVICE array pointer is mixed into
// the signature, so swapping the table produces a new keyed variant instead
// of a silent read of stale blocks. The view's CONTENTS remain free to
// change - they are device-side scalars.
//
// NO HOST BRANCHING ON DEVICE DATA. Between capture and replay the stage must
// not branch on anything a kernel produced. The submit path satisfies this by
// construction: it branches on the frame header (keyed) and nothing else, and
// the completion host-function stays OUTSIDE the captured region because
// cudaLaunchHostFunc is not capturable. The D2H token copy stays outside too:
// its destination is an unpinned host struct, which a capture would reject
// anyway, and stream order already guarantees the graph's writes are visible
// to it.
//
// DETECTED VIOLATIONS, NOT SILENT ONES. Capture can fail - an uncapturable
// call inside the driver launch, an unsupported stream, an empty recording -
// and every failure degrades to the eager launch the step would have made
// anyway, with the diagnostic on stderr. Eager is the definition of correct;
// a graph is only ever an accelerator of it. The one failure that must never
// degrade silently is a capture whose work DID NOT EXECUTE: stream capture
// records without running, so any path that abandons a capture re-issues the
// launch eagerly before returning.
//
// THREADING. The cache needs no lock: it is per pipeline slot, and the module
// hands a slot to exactly one submitter at a time. Two replays of one graph
// on one stream are stream-ordered against each other, so a slot's in-flight
// step and its next step cannot race.

#include "sparkpipe/spark_resident_decode_stage.h"

#include <stdint.h>

#define SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_EAGER 0u
#define SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_REPLAY 1u
#define SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_CAPTURE 2u

// Primary slot plus the spares the slot-state struct carries. The spare count
// is sized by the MTP draft variants, which is also the honest upper bound on
// simultaneously live decode shapes: batch bucket times speculation variant.
#define SPARK_RESIDENT_DECODE_STAGE_GRAPH_ENTRY_COUNT \
    (1u + SPARK_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT)

// The CUDA calls, behind a table so the decision logic below compiles - and
// runs - on a host with no driver. begin/end return 0 on success. end_capture
// ends the recording, instantiates it, and hands back the executable; on any
// failure it returns nonzero, has already destroyed whatever the recording
// produced, and the stream is capturing nothing. abort_capture ends and
// discards; it exists for the path where the recorded launch itself failed
// and instantiating the fragment would record a lie.
typedef struct SparkResidentDecodeStageGraphOps
{
    void *context;
    int32_t (*begin_capture)(void *context, void *cuda_stream);
    int32_t (*end_capture)(void *context, void *cuda_stream, void **exec_out);
    void (*abort_capture)(void *context, void *cuda_stream);
    int32_t (*launch)(void *context, void *exec, void *cuda_stream);
    void (*destroy)(void *context, void *exec);
} SparkResidentDecodeStageGraphOps;

// The driver launch sequence, as a callback so this file never learns what a
// layer is. During capture the callback RECORDS - its work does not execute -
// which is why every capture path that does not end in a replay re-invokes it.
typedef SparkStatus (*SparkResidentDecodeStageGraphLaunchFunction)(
    void *launch_context);

// FNV-1a over the host-derived inputs. A struct key would be exact, but the
// ABI fixes the slot-state layout at one uint64 per entry, so the key is a
// hash - and that is safe here for a reason a general cache could not claim:
// every input is a value the service itself holds constant for its lifetime
// (plan identities, the KV arena's device pointer) or a small integer from a
// bounded set (batch, stage counts, flags). The hash is a change detector
// over values that almost never change, not a partition of an adversarial
// space, and a collision requires two DIFFERENT live configurations to fold
// to the same 64-bit value while sharing one pipeline slot.
static inline uint64_t SparkResidentDecodeStageGraphSignatureMix(
    uint64_t hash,
    uint64_t value)
{
    return (hash ^ value) * 1099511628211ull;
}

static inline uint64_t SparkResidentDecodeStageGraphSpecializationSignature(
    uint32_t final_token_stage,
    uint32_t layer_count,
    const SparkResidentDecodeStageFrameContext *frame_context,
    const SparkKvBlockTableView *runtime_kv_block_table,
    const void *plan_identity)
{
    uint64_t hash;

    hash = 1469598103934665603ull;
    hash = SparkResidentDecodeStageGraphSignatureMix(hash, final_token_stage);
    hash = SparkResidentDecodeStageGraphSignatureMix(hash, layer_count);
    hash = SparkResidentDecodeStageGraphSignatureMix(
        hash,
        frame_context != 0 ? frame_context->flags : 0u);
    hash = SparkResidentDecodeStageGraphSignatureMix(
        hash,
        frame_context != 0 ? frame_context->logical_lane_count : 0u);
    hash = SparkResidentDecodeStageGraphSignatureMix(
        hash,
        frame_context != 0 ? frame_context->rows_per_lane : 0u);
    // The KV table's device array pointer, not the view's address: the view
    // is rebuilt per frame, the arena it points into is not. A table swap is
    // a new keyed variant; new table CONTENTS are the device-side scalars a
    // replay re-reads.
    hash = SparkResidentDecodeStageGraphSignatureMix(
        hash,
        (uint64_t)(uintptr_t)(runtime_kv_block_table != 0
            ? runtime_kv_block_table->physical_block_indices
            : 0));
    hash = SparkResidentDecodeStageGraphSignatureMix(
        hash,
        (uint64_t)(uintptr_t)plan_identity);
    return hash;
}

// A slot state starts zeroed from the context builder; this exists for tests
// and for any builder that reuses storage. It does NOT destroy live graphs -
// it cannot, the CUDA calls are not this file's - so a builder resetting a
// live state must destroy the execs first.
static inline void SparkResidentDecodeStageGraphCacheReset(
    SparkResidentDecodeStageCudaPipelineSlotState *slot_state)
{
    uint32_t index;

    slot_state->abi_version =
        SPARK_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION;
    slot_state->graph_active_sequence_count = 0u;
    slot_state->cuda_graph_exec = 0;
    slot_state->graph_specialization_signature = 0u;
    slot_state->graph_capture_count = 0u;
    for (index = 0u;
         index < SPARK_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT;
         ++index)
    {
        slot_state->cuda_graph_exec_cache[index] = 0;
        slot_state->graph_cache_active_sequence_counts[index] = 0u;
        slot_state->graph_cache_specialization_signatures[index] = 0u;
        slot_state->graph_cache_last_use_epochs[index] = 0u;
    }
    slot_state->graph_cache_clock = 0u;
    slot_state->graph_replay_count = 0u;
}

// Entry 0 is the primary (the struct's named fields); 1..N-1 index the spare
// arrays. The primary exists because the common case is one shape replayed
// for thousands of steps, and that lookup should be one comparison.
static inline void *SparkResidentDecodeStageGraphCacheEntryExec(
    const SparkResidentDecodeStageCudaPipelineSlotState *slot_state,
    int32_t entry)
{
    if (entry == 0)
    {
        return slot_state->cuda_graph_exec;
    }
    return slot_state->cuda_graph_exec_cache[entry - 1];
}

static inline int32_t SparkResidentDecodeStageGraphCacheFind(
    SparkResidentDecodeStageCudaPipelineSlotState *slot_state,
    uint32_t active_sequence_count,
    uint64_t specialization_signature)
{
    uint32_t index;

    if (slot_state->cuda_graph_exec != 0 &&
        slot_state->graph_active_sequence_count == active_sequence_count &&
        slot_state->graph_specialization_signature ==
            specialization_signature)
    {
        return 0;
    }
    for (index = 0u;
         index < SPARK_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT;
         ++index)
    {
        if (slot_state->cuda_graph_exec_cache[index] != 0 &&
            slot_state->graph_cache_active_sequence_counts[index] ==
                active_sequence_count &&
            slot_state->graph_cache_specialization_signatures[index] ==
                specialization_signature)
        {
            slot_state->graph_cache_clock += 1u;
            slot_state->graph_cache_last_use_epochs[index] =
                slot_state->graph_cache_clock;
            return (int32_t)(index + 1u);
        }
    }
    return -1;
}

// Remove an entry whose exec proved unusable (a replay launch that failed).
// The caller destroys the returned exec through the ops table; the slot is
// left empty for the next capture.
static inline void *SparkResidentDecodeStageGraphCacheEvict(
    SparkResidentDecodeStageCudaPipelineSlotState *slot_state,
    int32_t entry)
{
    void *exec;

    if (entry < 0)
    {
        return 0;
    }
    if (entry == 0)
    {
        exec = slot_state->cuda_graph_exec;
        slot_state->cuda_graph_exec = 0;
        slot_state->graph_active_sequence_count = 0u;
        slot_state->graph_specialization_signature = 0u;
        return exec;
    }
    exec = slot_state->cuda_graph_exec_cache[entry - 1];
    slot_state->cuda_graph_exec_cache[entry - 1] = 0;
    slot_state->graph_cache_active_sequence_counts[entry - 1] = 0u;
    slot_state->graph_cache_specialization_signatures[entry - 1] = 0u;
    slot_state->graph_cache_last_use_epochs[entry - 1] = 0u;
    return exec;
}

// Insert a fresh capture as the new primary. A live primary is demoted to the
// coldest spare; a live occupant of THAT slot is returned so the caller can
// destroy it through the ops table (destroying a graph exec is a driver call
// this file cannot make). LRU rather than FIFO because decode shapes recur:
// the B64 shape evicted for a one-off B1 prefill-adjacent variant is the
// shape the next hundred steps want back.
static inline void *SparkResidentDecodeStageGraphCacheStore(
    SparkResidentDecodeStageCudaPipelineSlotState *slot_state,
    uint32_t active_sequence_count,
    uint64_t specialization_signature,
    void *exec)
{
    void *evicted;

    evicted = 0;
    if (slot_state->cuda_graph_exec != 0)
    {
        uint32_t victim;
        uint32_t index;
        uint64_t oldest_epoch;

        victim = SPARK_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT;
        oldest_epoch = UINT64_MAX;
        for (index = 0u;
             index < SPARK_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT;
             ++index)
        {
            if (slot_state->cuda_graph_exec_cache[index] == 0)
            {
                victim = index;
                break;
            }
            if (slot_state->graph_cache_last_use_epochs[index] <
                oldest_epoch)
            {
                oldest_epoch =
                    slot_state->graph_cache_last_use_epochs[index];
                victim = index;
            }
        }
        evicted = slot_state->cuda_graph_exec_cache[victim];
        slot_state->cuda_graph_exec_cache[victim] =
            slot_state->cuda_graph_exec;
        slot_state->graph_cache_active_sequence_counts[victim] =
            slot_state->graph_active_sequence_count;
        slot_state->graph_cache_specialization_signatures[victim] =
            slot_state->graph_specialization_signature;
        slot_state->graph_cache_clock += 1u;
        slot_state->graph_cache_last_use_epochs[victim] =
            slot_state->graph_cache_clock;
    }
    slot_state->cuda_graph_exec = exec;
    slot_state->graph_active_sequence_count = active_sequence_count;
    slot_state->graph_specialization_signature = specialization_signature;
    slot_state->graph_capture_count += 1u;
    return evicted;
}

// The whole decision. Returns the step's status; mode_out (when non-null)
// reports which path ran so the snapshot can distinguish a replay storm from
// a capture storm. slot_state may be null - that IS the eager path, and the
// eligibility check that produced the null lives at the call site where the
// node context is.
static inline SparkStatus SparkResidentDecodeStageGraphSubmit(
    SparkResidentDecodeStageCudaPipelineSlotState *slot_state,
    uint32_t active_sequence_count,
    uint64_t specialization_signature,
    const SparkResidentDecodeStageGraphOps *ops,
    void *cuda_stream,
    SparkResidentDecodeStageGraphLaunchFunction launch_function,
    void *launch_context,
    uint32_t *mode_out)
{
    int32_t entry;
    void *exec;
    void *evicted;
    SparkStatus status;

    if (mode_out != 0)
    {
        *mode_out = SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_EAGER;
    }
    if (launch_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (slot_state == 0 || ops == 0 ||
        ops->begin_capture == 0 || ops->end_capture == 0 ||
        ops->abort_capture == 0 || ops->launch == 0 || ops->destroy == 0)
    {
        return launch_function(launch_context);
    }

    entry = SparkResidentDecodeStageGraphCacheFind(
        slot_state,
        active_sequence_count,
        specialization_signature);
    if (entry >= 0)
    {
        exec = SparkResidentDecodeStageGraphCacheEntryExec(slot_state, entry);
        if (ops->launch(ops->context, exec, cuda_stream) == 0)
        {
            slot_state->graph_replay_count += 1u;
            if (mode_out != 0)
            {
                *mode_out = SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_REPLAY;
            }
            return SPARK_STATUS_OK;
        }
        // A recording that will not launch poisons every future step with
        // this shape. Drop it and run eager; the next step recaptures.
        ops->destroy(
            ops->context,
            SparkResidentDecodeStageGraphCacheEvict(slot_state, entry));
        return launch_function(launch_context);
    }

    if (ops->begin_capture(ops->context, cuda_stream) != 0)
    {
        // Capture unsupported on this stream or device: the step runs as it
        // always has. Not an error, not retried noisily - the miss path pays
        // one failed begin per step, which is the price of keeping the
        // fallback stateless.
        return launch_function(launch_context);
    }
    status = launch_function(launch_context);
    if (status != SPARK_STATUS_OK)
    {
        // The launch failed mid-recording. The recording is a fragment of a
        // step and the stream is still capturing; end and discard both. The
        // recorded work never executed, and the caller already has the
        // failure, so there is nothing to re-run here.
        ops->abort_capture(ops->context, cuda_stream);
        return status;
    }
    if (ops->end_capture(ops->context, cuda_stream, &exec) != 0)
    {
        // The recording could not become a graph - an uncapturable call
        // inside the driver launch is the expected cause. Its work never
        // executed, so the step still owes the GPU everything: re-run eager.
        return launch_function(launch_context);
    }
    evicted = SparkResidentDecodeStageGraphCacheStore(
        slot_state,
        active_sequence_count,
        specialization_signature,
        exec);
    if (evicted != 0)
    {
        ops->destroy(ops->context, evicted);
    }
    if (ops->launch(ops->context, exec, cuda_stream) != 0)
    {
        // Instantiated but will not launch; nothing has executed this step.
        ops->destroy(
            ops->context,
            SparkResidentDecodeStageGraphCacheEvict(slot_state, 0));
        return launch_function(launch_context);
    }
    if (mode_out != 0)
    {
        *mode_out = SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_CAPTURE;
    }
    return SPARK_STATUS_OK;
}

// Eligibility, decided where the node context is visible. Graph replay is
// opt-in per context (validation rejects the REQUIRE flag without it) and
// requires the per-slot state array the builder allocates alongside it.
static inline SparkResidentDecodeStageCudaPipelineSlotState *
SparkResidentDecodeStageGraphSlotState(
    const SparkResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index)
{
    if (node_context == 0 ||
        node_context->enable_cuda_graph_replay == 0u ||
        node_context->cuda_pipeline_slot_states == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count)
    {
        return 0;
    }
    return &node_context->cuda_pipeline_slot_states[pipeline_slot_index];
}

#endif

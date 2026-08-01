// The stage-side CUDA graph decision logic, driven on a host with no driver.
//
// inference/stage/graph_replay.h owns the key, the fixed-slot cache, and the
// capture/replay/fallback decision; dispatch.cu owns five thin CUDA calls.
// This test swaps those five for a recorder and checks every branch the hot
// path can take: the first-sighting capture, the steady-state replay, the
// keyed variants, the LRU eviction, and the three failure degradations -
// capture unsupported, capture void (whose recorded work never executed and
// must be re-run eager), and a replay that will not launch.

#include "inference/stage/graph_replay.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define MOCK_LOG_CAPACITY 64u

typedef struct MockGraph
{
    char log[MOCK_LOG_CAPACITY];
    uint32_t log_count;
    uint32_t driver_calls;
    uint32_t destroyed_execs[8u];
    uint32_t destroyed_count;
    uint32_t exec_serial;
    uint32_t launched_exec;
    int32_t begin_result;
    int32_t end_result;
    int32_t launch_result;
    SparkStatus driver_status;
} MockGraph;

static void MockNote(MockGraph *mock, char event)
{
    assert(mock->log_count < MOCK_LOG_CAPACITY);
    mock->log[mock->log_count] = event;
    mock->log_count += 1u;
    mock->log[mock->log_count] = '\0';
}

static int32_t MockBeginCapture(void *context, void *cuda_stream)
{
    MockGraph *mock = (MockGraph *)context;
    (void)cuda_stream;
    MockNote(mock, 'b');
    return mock->begin_result;
}

static int32_t MockEndCapture(void *context, void *cuda_stream, void **exec_out)
{
    MockGraph *mock = (MockGraph *)context;
    (void)cuda_stream;
    MockNote(mock, 'e');
    if (mock->end_result != 0)
    {
        return mock->end_result;
    }
    mock->exec_serial += 1u;
    *exec_out = (void *)(uintptr_t)mock->exec_serial;
    return 0;
}

static void MockAbortCapture(void *context, void *cuda_stream)
{
    MockGraph *mock = (MockGraph *)context;
    (void)cuda_stream;
    MockNote(mock, 'a');
}

static int32_t MockLaunch(void *context, void *exec, void *cuda_stream)
{
    MockGraph *mock = (MockGraph *)context;
    (void)cuda_stream;
    MockNote(mock, 'r');
    mock->launched_exec = (uint32_t)(uintptr_t)exec;
    return mock->launch_result;
}

static void MockDestroy(void *context, void *exec)
{
    MockGraph *mock = (MockGraph *)context;
    MockNote(mock, 'd');
    assert(mock->destroyed_count < 8u);
    mock->destroyed_execs[mock->destroyed_count] =
        (uint32_t)(uintptr_t)exec;
    mock->destroyed_count += 1u;
}

static SparkStatus MockDriverLaunch(void *launch_context)
{
    MockGraph *mock = (MockGraph *)launch_context;
    MockNote(mock, 'L');
    mock->driver_calls += 1u;
    return mock->driver_status;
}

static const SparkResidentDecodeStageGraphOps MOCK_OPS = {
    0,
    MockBeginCapture,
    MockEndCapture,
    MockAbortCapture,
    MockLaunch,
    MockDestroy
};

static void MockReset(MockGraph *mock)
{
    memset(mock, 0, sizeof(*mock));
    mock->driver_status = SPARK_STATUS_OK;
}

static SparkStatus MockSubmit(
    MockGraph *mock,
    SparkResidentDecodeStageCudaPipelineSlotState *slot_state,
    uint32_t active,
    uint64_t signature,
    uint32_t *mode_out)
{
    SparkResidentDecodeStageGraphOps ops;

    ops = MOCK_OPS;
    ops.context = mock;
    return SparkResidentDecodeStageGraphSubmit(
        slot_state,
        active,
        signature,
        &ops,
        (void *)0x1,
        MockDriverLaunch,
        mock,
        mode_out);
}

static void TestEagerWithoutSlotState(void)
{
    MockGraph mock;
    uint32_t mode;

    MockReset(&mock);
    mode = 99u;
    assert(MockSubmit(&mock, 0, 8u, 0xabcu, &mode) == SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_EAGER);
    assert(mock.driver_calls == 1u);
    assert(strcmp(mock.log, "L") == 0);
}

static void TestCaptureThenReplay(void)
{
    MockGraph mock;
    SparkResidentDecodeStageCudaPipelineSlotState slot_state;
    uint32_t mode;

    MockReset(&mock);
    memset(&slot_state, 0, sizeof(slot_state));
    SparkResidentDecodeStageGraphCacheReset(&slot_state);

    // First sighting of the shape: record the driver launch, instantiate,
    // then launch the fresh graph so the captured step still executes.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_CAPTURE);
    assert(strcmp(mock.log, "bLer") == 0);
    assert(mock.launched_exec == 1u);
    assert(slot_state.graph_capture_count == 1u);
    assert(slot_state.graph_replay_count == 0u);

    // Same shape: one submission, no host launch work at all.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_REPLAY);
    assert(strcmp(mock.log, "bLerr") == 0);
    assert(mock.driver_calls == 1u);
    assert(slot_state.graph_replay_count == 1u);
}

static void TestKeyedVariantsAndEviction(void)
{
    MockGraph mock;
    SparkResidentDecodeStageCudaPipelineSlotState slot_state;
    uint32_t mode;
    uint32_t key;
    uint32_t spare_capacity;

    MockReset(&mock);
    memset(&slot_state, 0, sizeof(slot_state));
    SparkResidentDecodeStageGraphCacheReset(&slot_state);
    spare_capacity =
        SPARK_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT;

    // A different batch is a different graph: the active count is a grid
    // dimension baked into the recording.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(MockSubmit(&mock, &slot_state, 64u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_CAPTURE);
    assert(mock.driver_calls == 2u);
    // The B8 variant survives beside B64: replaying it costs no capture.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_REPLAY);
    assert(mock.driver_calls == 2u);

    // Fill every remaining entry with distinct signatures, then add one
    // more: the coldest spare is evicted and destroyed through the ops
    // table, never silently dropped. Two variants are live (B8 and B64),
    // so spare_capacity - 1 more exactly fills the cache.
    for (key = 0u; key + 1u < spare_capacity; ++key)
    {
        assert(MockSubmit(&mock, &slot_state, 8u, 0x1000u + key, &mode) ==
            SPARK_STATUS_OK);
    }
    assert(mock.destroyed_count == 0u);
    assert(MockSubmit(&mock, &slot_state, 16u, 0x999u, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_CAPTURE);
    assert(mock.destroyed_count == 1u);
    // The evicted exec is the least-recently-used: the B8 graph (exec
    // serial 1), demoted first and replayed once before every fill.
    assert(mock.destroyed_execs[0] == 1u);
}

static void TestBeginFailureFallsBackEager(void)
{
    MockGraph mock;
    SparkResidentDecodeStageCudaPipelineSlotState slot_state;
    uint32_t mode;

    MockReset(&mock);
    mock.begin_result = -1;
    memset(&slot_state, 0, sizeof(slot_state));
    SparkResidentDecodeStageGraphCacheReset(&slot_state);
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_EAGER);
    assert(strcmp(mock.log, "bL") == 0);
    assert(mock.driver_calls == 1u);
}

static void TestVoidCaptureRerunsEager(void)
{
    MockGraph mock;
    SparkResidentDecodeStageCudaPipelineSlotState slot_state;
    uint32_t mode;

    MockReset(&mock);
    mock.end_result = -1;
    memset(&slot_state, 0, sizeof(slot_state));
    SparkResidentDecodeStageGraphCacheReset(&slot_state);
    // The recording was discarded: its launches never executed, so the
    // driver launch must run a second time, for real.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_EAGER);
    assert(strcmp(mock.log, "bLeL") == 0);
    assert(mock.driver_calls == 2u);
    assert(slot_state.graph_capture_count == 0u);
}

static void TestDriverErrorAbortsCapture(void)
{
    MockGraph mock;
    SparkResidentDecodeStageCudaPipelineSlotState slot_state;
    uint32_t mode;

    MockReset(&mock);
    mock.driver_status = SPARK_STATUS_INTERNAL_ERROR;
    memset(&slot_state, 0, sizeof(slot_state));
    SparkResidentDecodeStageGraphCacheReset(&slot_state);
    // The launch failed mid-recording: the fragment is discarded and the
    // error propagates. Nothing re-runs - the caller owns the failure.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_INTERNAL_ERROR);
    assert(strcmp(mock.log, "bLa") == 0);
    assert(slot_state.graph_capture_count == 0u);
}

static void TestFailedReplayIsEvictedAndEager(void)
{
    MockGraph mock;
    SparkResidentDecodeStageCudaPipelineSlotState slot_state;
    uint32_t mode;

    MockReset(&mock);
    memset(&slot_state, 0, sizeof(slot_state));
    SparkResidentDecodeStageGraphCacheReset(&slot_state);
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    mock.launch_result = -1;
    // A recording that will not launch is worse than none: it is destroyed
    // and the step runs eager. The next step recaptures.
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_EAGER);
    assert(strcmp(mock.log, "bLerrdL") == 0);
    assert(mock.destroyed_execs[0] == 1u);
    assert(slot_state.cuda_graph_exec == 0);
    mock.launch_result = 0;
    assert(MockSubmit(&mock, &slot_state, 8u, 0xabcu, &mode) ==
        SPARK_STATUS_OK);
    assert(mode == SPARK_RESIDENT_DECODE_STAGE_GRAPH_MODE_CAPTURE);
}

static void TestSignatureCoversEveryHostDerivedInput(void)
{
    SparkResidentDecodeStageFrameContext frame_context;
    SparkKvBlockTableView kv_table;
    uint64_t base;

    memset(&frame_context, 0, sizeof(frame_context));
    memset(&kv_table, 0, sizeof(kv_table));
    frame_context.logical_lane_count = 8u;
    frame_context.rows_per_lane = 1u;
    kv_table.physical_block_indices = (const uint32_t *)0xfeed0000u;
    base = SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 61u, &frame_context, &kv_table, (const void *)0x1234u);
    // Same inputs, same key - replay stability.
    assert(base == SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 61u, &frame_context, &kv_table, (const void *)0x1234u));
    // Each host-derived input moves the key. The one that matters most is
    // the KV table pointer: a table swap without a new key is a replay that
    // attends to another sequence's blocks.
    assert(base != SparkResidentDecodeStageGraphSpecializationSignature(
        0u, 61u, &frame_context, &kv_table, (const void *)0x1234u));
    assert(base != SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 62u, &frame_context, &kv_table, (const void *)0x1234u));
    frame_context.flags =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS;
    assert(base != SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 61u, &frame_context, &kv_table, (const void *)0x1234u));
    frame_context.flags = 0u;
    frame_context.rows_per_lane = 2u;
    assert(base != SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 61u, &frame_context, &kv_table, (const void *)0x1234u));
    frame_context.rows_per_lane = 1u;
    kv_table.physical_block_indices = (const uint32_t *)0xbeaf0000u;
    assert(base != SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 61u, &frame_context, &kv_table, (const void *)0x1234u));
    kv_table.physical_block_indices = (const uint32_t *)0xfeed0000u;
    assert(base != SparkResidentDecodeStageGraphSpecializationSignature(
        1u, 61u, &frame_context, &kv_table, (const void *)0x5678u));
    // A null frame context and a null table are representable, not crashes:
    // intermediate ring stages submit decode slices with neither.
    (void)SparkResidentDecodeStageGraphSpecializationSignature(
        0u, 8u, 0, 0, 0);
}

int main(void)
{
    TestEagerWithoutSlotState();
    TestCaptureThenReplay();
    TestKeyedVariantsAndEviction();
    TestBeginFailureFallsBackEager();
    TestVoidCaptureRerunsEager();
    TestDriverErrorAbortsCapture();
    TestFailedReplayIsEvictedAndEager();
    TestSignatureCoversEveryHostDerivedInput();
    printf("stage graph replay: capture, replay, keyed variants, LRU "
        "eviction, and every eager fallback verified\n");
    return 0;
}

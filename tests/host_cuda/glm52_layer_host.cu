// Run the real GLM 5.2 layer on a CPU: one attention, one MoE, one dense MLP
// in both gate/up forms, the head, and the bind-time gate/up contiguity
// decision - the code that ships, end to end, two rows.
//
// The glm5_2 driver had no harness at all: the per-kernel tests cover kernels,
// and the python contracts read the launch sites, but nothing EXECUTED the
// layer. That is how the join-assembled KV slot and the head's dropped
// residual could sit in the tree unobserved - both live between the kernels,
// which is exactly the span only a layer harness sees. k3_slice_host found six
// defects in that span for K3; this is the same instrument for GLM.
//
// The GEMM is the recorder - output 0.125 * call index, constant across rows,
// honouring the output row stride and column offset - so the whole stream is
// closed-form arithmetic the checker recomputes: the norm chain, the rope
// rotations, the slot layout written directly by the kv projections, the
// attention softmax over four positions, the counting sort, the gather, and
// the final norm over hidden + residual. Everything else is the shipping
// kernel.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/tile.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "runtime/gemm.cuh"
std::vector<LmRecordedGemm> lm_recorded_gemms;

// kv.cuh guards its store kernel on __CUDACC__; scope the macro to that one
// include exactly as k3_slice_host.cu does, so dtype.cuh never sees it.
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

#include "inference/llms/glm5_2/layer.cuh"
#include "inference/llms/glm5_2/bind.cu"

#define ROWS 2u
#define PACKED (ROWS * GLM52_TOP_K)
#define CONTEXT 4u
#define POSITION 3u
#define SEQUENCES 2u
#define HEAD_VOCAB 128u
#define QK_SCALE 0.05f

// The dead outputs of the retired join path stay null on purpose: a regression
// that routes the kv projections through them again crashes the recorder here
// instead of silently re-adding the launch on hardware.
static uint16_t hidden[ROWS * GLM52_HIDDEN];
static uint16_t residual[ROWS * GLM52_HIDDEN];
static uint16_t normed[ROWS * GLM52_HIDDEN];
static uint16_t query_latent[ROWS * GLM52_ATTN_HEADS * GLM52_LATENT];
static uint16_t query_rope[ROWS * GLM52_ATTN_HEADS * GLM52_ROPE_DIM];
static uint16_t kv_slot[ROWS * GLM52_LATENT_ROW];
static uint16_t attention_latent[ROWS * GLM52_ATTN_HEADS * GLM52_LATENT];
static uint16_t attention_out[ROWS * GLM52_HIDDEN];
static uint16_t gate_up[PACKED * GLM52_GATE_UP_DIM];
static uint16_t intermediate[PACKED * GLM52_EXPERT_INTERMEDIATE];
static uint16_t expert_out[PACKED * GLM52_HIDDEN];
static uint16_t gather_check[PACKED * GLM52_HIDDEN];
static float router_logits[ROWS * GLM52_EXPERTS];
static uint32_t route_expert[PACKED];
static float route_weight[PACKED];
static uint32_t route_source_token[PACKED];
static uint32_t route_packed_row[PACKED];
static float head_candidate_score[ROWS];
static uint32_t head_candidate_token[ROWS];
static uint32_t output_token[ROWS];
static float output_score[ROWS];
static uint32_t group_row_offset[GLM52_EXPERTS + 1u];
static uint32_t group_tile_prefix_w1[GLM52_EXPERTS + 1u];
static uint32_t group_tile_prefix_w2[GLM52_EXPERTS + 1u];
static uint32_t dense_row_offset[ROWS + 1u];
static uint32_t dense_tile_prefix[ROWS + 1u];

static uint8_t kv_pool[SEQUENCES * Glm52Kv::kPageBytes];
static uint32_t page_table[SEQUENCES];
static uint32_t sequence_of_row[ROWS];
static uint32_t context_length[SEQUENCES];
static uint32_t positions[ROWS];
static LmKvAccessError kv_access_error;

static uint16_t attn_norm_w[GLM52_HIDDEN];
static uint16_t mlp_norm_w[GLM52_HIDDEN];
static uint16_t head_norm_w[GLM52_HIDDEN];
static uint16_t head_weight[HEAD_VOCAB * GLM52_HIDDEN];

// Distinct addresses, never read: the recorder logs the pointer, so each
// weight must be its own buffer for the gemm log to name it.
static uint16_t w_q_latent[8], w_q_rope[8], w_k_rope[8], w_kv_latent[8];
static uint16_t w_o[8], w_router[8], w_down[8], w_e1[8], w_e2[8];
static float s_e1[8], s_e2[8];

static uint32_t seed = 13579u;
static float NextRandom(void)
{
    seed = (seed * 1664525u) + 1013904223u;
    return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static void FillRandom(uint16_t *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        values[index] = LmFloatToBf16(NextRandom());
}

static void FillNormWeight(uint16_t *values, uint32_t count)
{
    uint32_t index;
    for (index = 0u; index < count; ++index)
        values[index] = LmFloatToBf16(1.0f + 0.2f * NextRandom());
}

static void Emit(const char *tag, const uint16_t *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        printf("%s %.9g\n", tag, (double)LmBf16ToFloat(values[index]));
}

static void EmitSample(const char *tag, const uint16_t *values, uint64_t rows,
    uint64_t row_width, uint64_t stride)
{
    uint64_t row, index;
    for (row = 0u; row < rows; ++row)
        for (index = 0u; index < row_width; index += stride)
            printf("%s %.9g\n", tag,
                (double)LmBf16ToFloat(values[(row * row_width) + index]));
}

static void EmitU32(const char *tag, const uint32_t *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        printf("%s %u\n", tag, values[index]);
}

static void EmitF32(const char *tag, const float *values, uint64_t count)
{
    uint64_t index;
    for (index = 0u; index < count; ++index)
        printf("%s %.9g\n", tag, (double)values[index]);
}

static const char *PointerName(const void *pointer)
{
    if (pointer == normed) return "normed";
    if (pointer == query_latent) return "query_latent";
    if (pointer == query_rope) return "query_rope";
    if (pointer == kv_slot) return "kv_slot";
    if (pointer == attention_latent) return "attention_latent";
    if (pointer == attention_out) return "attention_out";
    if (pointer == gate_up) return "gate_up";
    if (pointer == intermediate) return "intermediate";
    if (pointer == expert_out) return "expert_out";
    if (pointer == hidden) return "hidden";
    if (pointer == router_logits) return "router_logits";
    if (pointer == w_q_latent) return "w_q_latent";
    if (pointer == w_q_rope) return "w_q_rope";
    if (pointer == w_k_rope) return "w_k_rope";
    if (pointer == w_kv_latent) return "w_kv_latent";
    if (pointer == w_o) return "w_o";
    if (pointer == w_router) return "w_router";
    if (pointer == w_down) return "w_down";
    if (pointer == w_e1) return "w_e1";
    if (pointer == w_e2) return "w_e2";
    return "unknown";
}

static void EmitGemmLog(uint32_t begin, const char *phase)
{
    uint32_t index;
    for (index = begin; index < lm_recorded_gemms.size(); ++index)
    {
        const LmRecordedGemm *record = &lm_recorded_gemms[index];
        printf("gemm %s %u K %u N %u rows %u grouped %u act %s w %s dst %s\n",
            phase, index + 1u, record->input_dimension,
            record->output_dimension, record->packed_rows,
            record->grouped ? 1u : 0u, PointerName(record->activation),
            PointerName(record->weight), PointerName(record->output));
    }
}

static uint32_t CountPoison(const uint16_t *values, uint64_t count, uint16_t poison)
{
    uint64_t index;
    uint32_t found = 0u;
    for (index = 0u; index < count; ++index)
        if (values[index] == poison)
            ++found;
    return found;
}

int main(void)
{
    Glm52LayerBuffers buffers;
    uint32_t row, index;
    int32_t status;
    uint16_t poison;

    memset(&buffers, 0, sizeof(buffers));
    LmKvAccessErrorReset(&kv_access_error);
    FillRandom(hidden, ROWS * GLM52_HIDDEN);
    FillRandom(residual, ROWS * GLM52_HIDDEN);
    FillNormWeight(attn_norm_w, GLM52_HIDDEN);
    FillNormWeight(mlp_norm_w, GLM52_HIDDEN);
    FillNormWeight(head_norm_w, GLM52_HIDDEN);
    FillRandom(head_weight, (uint64_t)HEAD_VOCAB * GLM52_HIDDEN);

    // Three cached positions per sequence, filled before the step; the step
    // stores position 3 and attends over all four.
    for (row = 0u; row < SEQUENCES; ++row)
    {
        uint32_t position, element;
        page_table[row] = row;
        context_length[row] = CONTEXT;
        for (position = 0u; position < POSITION; ++position)
        {
            uint16_t *slot = (uint16_t *)(kv_pool +
                (row * Glm52Kv::kPageBytes) +
                (position * Glm52Kv::kSlotBytes));
            for (element = 0u; element < GLM52_LATENT_ROW; ++element)
                slot[element] = LmFloatToBf16(NextRandom());
        }
    }
    for (row = 0u; row < ROWS; ++row)
    {
        sequence_of_row[row] = row;
        positions[row] = POSITION;
    }
    dense_row_offset[0] = 0u;
    dense_row_offset[1] = ROWS;
    dense_row_offset[2] = ROWS;

    buffers.dense_row_offset = dense_row_offset;
    buffers.dense_tile_prefix = dense_tile_prefix;
    buffers.attn_norm_weight = attn_norm_w;
    buffers.absorbed.query_latent_weight = w_q_latent;
    buffers.absorbed.query_rope_weight = w_q_rope;
    buffers.absorbed.key_rope_weight = w_k_rope;
    buffers.absorbed.kv_latent_weight = w_kv_latent;
    buffers.use_absorbed = true;
    buffers.qk_scale = QK_SCALE;
    buffers.output_weight = w_o;
    buffers.mlp_norm_weight = mlp_norm_w;
    buffers.router_weight = w_router;
    buffers.expert_w1_weight = w_e1;
    buffers.expert_w1_scale = s_e1;
    buffers.expert_w2_weight = w_e2;
    buffers.expert_w2_scale = s_e2;
    buffers.hidden_bf16 = hidden;
    buffers.residual_bf16 = residual;
    buffers.normed_bf16 = normed;
    buffers.projected.query_latent_bf16 = query_latent;
    buffers.projected.query_rope_bf16 = query_rope;
    buffers.kv_slot_bf16 = kv_slot;
    buffers.attention_latent_bf16 = attention_latent;
    buffers.attention_out_bf16 = attention_out;
    buffers.gate_up_bf16 = gate_up;
    buffers.intermediate_bf16 = intermediate;
    buffers.expert_out_bf16 = expert_out;
    buffers.router_logits = router_logits;
    buffers.route_expert = route_expert;
    buffers.route_weight = route_weight;
    buffers.route_source_token = route_source_token;
    buffers.route_packed_row = route_packed_row;
    buffers.head_candidate_score = head_candidate_score;
    buffers.head_candidate_token = head_candidate_token;
    buffers.output_token = output_token;
    buffers.output_score = output_score;
    buffers.group_row_offset = group_row_offset;
    buffers.group_tile_prefix_w1 = group_tile_prefix_w1;
    buffers.group_tile_prefix_w2 = group_tile_prefix_w2;
    buffers.cache.pool = kv_pool;
    buffers.cache.page_table = page_table;
    buffers.cache.page_table_stride = 1u;
    buffers.cache.sequence_count = SEQUENCES;
    buffers.cache.pool_page_count = SEQUENCES;
    buffers.cache.access_error = &kv_access_error;
    buffers.sequence_of_row = sequence_of_row;
    buffers.context_length = context_length;
    buffers.positions = positions;
    buffers.row_positions = 0;

    printf("qkscale %.9g\n", (double)QK_SCALE);
    printf("theta %.9g\n", (double)GLM52_ROPE_THETA);
    printf("routedscale %.9g\n", (double)GLM52_ROUTED_SCALE);
    printf("eps %.9g\n", (double)GLM52_RMS_EPSILON);

    status = Glm52LayerAttention(&buffers, ROWS, CONTEXT, 0u, 48u, 0);
    printf("status attention %d\n", (int)status);
    EmitGemmLog(0u, "attention");
    Emit("normed1", normed, ROWS * GLM52_HIDDEN);
    Emit("kvslot", kv_slot, ROWS * GLM52_LATENT_ROW);
    for (row = 0u; row < SEQUENCES; ++row)
    {
        const uint16_t *slot = (const uint16_t *)(kv_pool +
            (row * Glm52Kv::kPageBytes) + (POSITION * Glm52Kv::kSlotBytes));
        Emit("slot", slot, GLM52_LATENT_ROW);
    }
    // The layer's attention decode runs at GLM52_ATTN_THREADS, which the shim
    // floors at 64: on the host it writes only every sixty-fourth output
    // element from a partially-staged query. The recorder erases that output
    // before anything reads it, so the harness checks the kernel's math at a
    // small geometry below instead, where one thread covers everything.
    {
        using TestKv = LmKvLatent<16u, 8u, 8u, 64u>;
        static uint8_t test_pool[TestKv::kPageBytes];
        static uint32_t test_pages[1];
        static uint16_t test_query_latent[2 * 8];
        static uint16_t test_query_rope[2 * 8];
        static uint16_t test_out[2 * 8];
        static uint32_t test_sequence[1];
        static uint32_t test_context[1];
        LmKvView test_cache;
        LmKvAccessError test_access_error;
        uint32_t head, position, element;

        test_pages[0] = 0u;
        test_sequence[0] = 0u;
        test_context[0] = 3u;
        LmKvAccessErrorReset(&test_access_error);
        test_cache.pool = test_pool;
        test_cache.page_table = test_pages;
        test_cache.page_table_stride = 1u;
        test_cache.sequence_count = 1u;
        test_cache.pool_page_count = 1u;
        test_cache.access_error = &test_access_error;
        for (position = 0u; position < 3u; ++position)
            for (element = 0u; element < 16u; ++element)
                ((uint16_t *)(test_pool + (position * TestKv::kSlotBytes)))
                    [element] = LmFloatToBf16(
                        0.1f * (float)(position + 1u) +
                        0.01f * (float)element);
        for (head = 0u; head < 2u; ++head)
            for (element = 0u; element < 8u; ++element)
            {
                test_query_latent[(head * 8u) + element] = LmFloatToBf16(
                    0.3f - (0.02f * (float)element));
                test_query_rope[(head * 8u) + element] = LmFloatToBf16(
                    -0.2f + (0.03f * (float)element) +
                    (0.05f * (float)head));
            }
        LM_HOST_LAUNCH(
            dim3(1u, 2u),
            (LmLatentAttentionDecodeKernel<TestKv, 1u, 8u, 8u>(
                test_query_latent, test_query_rope, test_cache,
                test_sequence, test_context, 0, 0u, 2u, 0.5f, test_out, 0)));
        Emit("smallattn", test_out, 2u * 8u);
    }

    status = Glm52LayerMoe(&buffers, ROWS, PACKED, 48u, 0);
    printf("status moe %d\n", (int)status);
    EmitGemmLog(5u, "moe");
    Emit("normed2", normed, ROWS * GLM52_HIDDEN);
    EmitU32("routeexpert", route_expert, PACKED);
    EmitF32("routeweight", route_weight, PACKED);
    EmitU32("routepacked", route_packed_row, PACKED);
    EmitU32("routesource", route_source_token, PACKED);
    EmitU32("groupoffset", group_row_offset, GLM52_EXPERTS + 1u);
    EmitU32("tileup", group_tile_prefix_w1, GLM52_EXPERTS + 1u);
    EmitU32("tiledown", group_tile_prefix_w2, GLM52_EXPERTS + 1u);
    EmitSample("inter", intermediate, PACKED, GLM52_EXPERT_INTERMEDIATE, 173u);
    Emit("hidden2", hidden, ROWS * GLM52_HIDDEN);

    // The gather is unobservable after w2 overwrote its destination, so rerun
    // it over the emitted route tables - same kernel, same arguments the layer
    // passed - and check the mapping itself.
    LM_HOST_LAUNCH(
        dim3(
            (GLM52_HIDDEN + GLM52_LAYER_THREADS - 1u) / GLM52_LAYER_THREADS,
            PACKED),
        (LmGatherRowsKernel<GLM52_LAYER_THREADS>(
            normed, route_source_token, gather_check, PACKED, GLM52_HIDDEN)));
    EmitSample("gather", gather_check, PACKED, GLM52_HIDDEN, 97u);

    // The dense MLP twice: once with the bind-time contiguity fact set, once
    // without. The poison fill turns a half-written gate_up into a failure
    // rather than a stale read.
    buffers.dense_gate_weight = w_q_latent;
    buffers.dense_up_weight = w_q_rope;
    buffers.dense_down_weight = w_down;
    buffers.dense_gate_up_fused = 1u;
    poison = LmFloatToBf16(-7.0f);
    memset(gate_up, 0, sizeof(gate_up));
    for (index = 0u; index < ROWS * GLM52_DENSE_INTERMEDIATE * 2u; ++index)
        gate_up[index] = poison;
    status = Glm52LayerDenseMlp(&buffers, ROWS, 48u, 0);
    printf("status densefused %d\n", (int)status);
    EmitGemmLog(8u, "densefused");
    printf("poison %u\n",
        CountPoison(gate_up, ROWS * GLM52_DENSE_INTERMEDIATE * 2u, poison));
    EmitSample("gateup", gate_up, ROWS, GLM52_DENSE_INTERMEDIATE * 2u, 997u);

    buffers.dense_gate_up_fused = 0u;
    for (index = 0u; index < ROWS * GLM52_DENSE_INTERMEDIATE * 2u; ++index)
        gate_up[index] = poison;
    status = Glm52LayerDenseMlp(&buffers, ROWS, 48u, 0);
    printf("status densetwo %d\n", (int)status);
    EmitGemmLog(10u, "densetwo");
    printf("poison %u\n",
        CountPoison(gate_up, ROWS * GLM52_DENSE_INTERMEDIATE * 2u, poison));
    EmitSample("gateup2", gate_up, ROWS, GLM52_DENSE_INTERMEDIATE * 2u, 997u);

    status = Glm52Head(&buffers, head_norm_w, head_weight, 0, HEAD_VOCAB,
        ROWS, 0);
    printf("status head %d\n", (int)status);
    Emit("normed3", normed, ROWS * GLM52_HIDDEN);
    EmitU32("token", output_token, ROWS);
    EmitF32("score", output_score, ROWS);

    // The bind-time decision itself: contiguous gate/up rows set the fusion
    // flag, anything else clears it. The up pointer is fabricated arithmetic
    // the layer never dereferences - the recorder logs pointers, not bytes.
    {
        SparkResidentDecodeStageNodeContext node;
        Glm52LayerBuffers bound;
        uintptr_t gate_address;

        memset(&node, 0, sizeof(node));
        memset(&bound, 0, sizeof(bound));
        node.layer_index = 0u;
        node.attention_norm_weight_bf16 = attn_norm_w;
        node.post_attention_norm_weight_bf16 = mlp_norm_w;
        node.attention_output_weight_bf16 = w_o;
        node.qk_scale = QK_SCALE;
        node.query_latent_weight_bf16 = w_q_latent;
        node.query_rope_weight_bf16 = w_q_rope;
        node.key_rope_weight_bf16 = w_k_rope;
        node.kv_latent_weight_bf16 = w_kv_latent;
        node.dense_gate_weight_bf16 = w_q_latent;
        node.dense_down_weight_bf16 = w_down;
        gate_address = (uintptr_t)w_q_latent;
        node.dense_up_weight_bf16 = (const void *)(gate_address +
            ((uintptr_t)GLM52_DENSE_INTERMEDIATE * GLM52_HIDDEN * 2u));
        status = Glm52BindLayer(&node, 0u, &buffers, &bound);
        printf("status bind %d\n", (int)status);
        printf("bindfused %u\n", bound.dense_gate_up_fused);
        node.dense_up_weight_bf16 = (const void *)(gate_address + 16u);
        status = Glm52BindLayer(&node, 0u, &buffers, &bound);
        printf("status bind2 %d\n", (int)status);
        printf("bindfused2 %u\n", bound.dense_gate_up_fused);
    }

    printf("done\n");
    return 0;
}

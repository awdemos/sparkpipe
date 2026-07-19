#ifndef SPARKPIPE_SPARK_QWEN36_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_QWEN36_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Qwen 3.6 27B resident decode stage, pipeline-parallel from version 1.
 *
 * Every driver in this family is a STAGE, never a whole model: a node owns a
 * contiguous layer slice [first_layer_index, first_layer_index + layer_count),
 * the stage count N is configuration (never a compiled constant), and the
 * hidden state crosses stage boundaries over the sparkpipe hidden transport.
 * Stage 0 additionally owns the embedding and consumes wire token ids; the
 * last stage owns the final norm and the LM head and produces sampled ids.
 * The whole-stack case is simply N == 1, not a separate mode.
 *
 * Why PP-N even for a model that fits one node: tokens/second is dominated by
 * per-stage compute, not the ~29us/hop ring latency (a decode microbatch of
 * 512 rows moves 512 x 10.24KB = 5.2MB per boundary, well under a millisecond
 * on the ring, fully overlapped with compute). Slicing the 54GB of weights
 * over N nodes turns almost the entire 128GB of every node into KV cache and
 * recurrent state: one node caps 512 resident lanes at roughly 2K tokens of
 * full-attention context each, while N = 13 holds the same 512 lanes at 32K+
 * with the pipeline kept full by the microbatch stream. Long-memory serving
 * at large concurrency is a memory problem first, and PP-N is the answer.
 *
 * What crosses a Qwen boundary: the residual hidden vector only, rows x 5120
 * bf16 per microbatch. GDN delta state, GDN conv tails and the paged KV cache
 * are resident on the stage that owns their layers and never move. There is
 * no sideband payload (contrast K3, whose AttnRes block array rides the
 * transport sideband, and DSv4, whose mHC streams quadruple the payload).
 *
 * Decode frames are BATCHED: one frame carries one next-token row for up to
 * max_active_sequence_count distinct lanes, which is what keeps every stage
 * of the pipeline saturated at 500-way long-memory concurrency. Prefill
 * frames remain one lane per frame, chunked at the GDN chunk width, because a
 * prefill already fills the stage on its own.
 */

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_QWEN36_MODEL_HIDDEN_DIMENSION
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_QWEN36_MODEL_LAYER_COUNT
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 32u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 512u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 2u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 3u

/*
 * One linear projection, bf16 or MXFP4 payload with per-group E8M0 scales.
 * Identical contract to the K3 view; the shared Linear kernel consumes both.
 */
typedef struct SparkQwen36LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkQwen36LinearView;

/*
 * Gated DeltaNet layer weights. The stage pack carries SPLIT projections
 * (query, key, value, gate) regardless of how the checkpoint fuses them; the
 * pack converter splits against the safetensors index at build time. The ba
 * projection produces the two per-value-head scalars (beta preactivation,
 * decay preactivation) as one 2 x 48 output. a_log and dt_bias are the fp32
 * per-value-head decay parameters. MODELING-PIN: the exact decay composition
 * (-exp(a_log) * softplus(a + dt_bias) in the Qwen3-Next lineage), the conv
 * bias presence, and the gated-norm weight shape are pinned against
 * modeling_qwen3_5 before the CUDA is written; each is one tensor row in the
 * shape table if the pin moves it.
 */
typedef struct SparkQwen36GdnLayerWeights
{
	SparkQwen36LinearView query;
	SparkQwen36LinearView key;
	SparkQwen36LinearView value;
	SparkQwen36LinearView gate;
	SparkQwen36LinearView ba;
	SparkQwen36LinearView output;
	const void *conv_weight_bf16;
	const void *conv_bias_bf16;
	const float *a_log_f32;
	const float *dt_bias_f32;
	const void *gdn_norm_weight_bf16;
} SparkQwen36GdnLayerWeights;

/*
 * Full attention layer weights. Query and key head RMSNorm gains span one
 * head (256). The output gate is a learned projection to the query width,
 * applied elementwise to the attention output before the output projection
 * (MODELING-PIN: gate activation).
 */
typedef struct SparkQwen36AttnLayerWeights
{
	SparkQwen36LinearView query;
	SparkQwen36LinearView key;
	SparkQwen36LinearView value;
	SparkQwen36LinearView output_gate;
	SparkQwen36LinearView output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SparkQwen36AttnLayerWeights;

typedef struct SparkQwen36FfnLayerWeights
{
	SparkQwen36LinearView gate;
	SparkQwen36LinearView up;
	SparkQwen36LinearView down;
} SparkQwen36FfnLayerWeights;

/*
 * GDN recurrent state for the stage's own GDN layers only. Two carried
 * pieces per lane per GDN layer: the dk x dv fp32 delta state per value head,
 * and the conv tail (the last kernel-1 columns of the concatenated q|k|v conv
 * input, bf16) that seeds the depthwise causal conv of the next dispatch.
 * gdn_layer_ordinal densely numbers the stage's GDN layers from zero, so a
 * slice that begins mid-period costs nothing. state_cold_by_row tells the
 * kernels to treat both pieces as zero on a lane's first touch.
 */
typedef struct SparkQwen36GdnStatePool
{
	uint32_t abi_version;
	uint32_t lane_capacity;
	uint32_t gdn_layer_count;
	uint32_t reserved0;
	float *state_f32;
	uint64_t state_lane_stride_elements;
	uint64_t state_layer_stride_elements;
	void *conv_tail_bf16;
	uint64_t conv_tail_lane_stride_elements;
	uint64_t conv_tail_layer_stride_elements;
	uint32_t *state_cold_by_row;
} SparkQwen36GdnStatePool;

/*
 * Paged KV cache table for the stage's own full-attention layers. One token
 * costs SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS bf16 elements per
 * full-attention layer (K then V, head-major, post-RoPE). Host mirrors are
 * required so the module can prove per-lane block coverage before a launch.
 */
typedef struct SparkQwen36KvBlockTableView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t lane_count;
	uint32_t lane_stride;
	uint32_t lane_capacity;
	const uint32_t *physical_block_indices;
	const uint32_t *lane_physical_block_counts;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_physical_block_counts;
} SparkQwen36KvBlockTableView;

/*
 * Per-slot device buffers, sized for max_active_sequence_count rows. A row is
 * one lane's next token in a decode microbatch, or one position of a prefill
 * chunk. hidden_input_bf16 is the transport landing buffer on stages other
 * than the first; hidden_output_bf16 is what the last layer of the slice
 * leaves for the send on stages other than the last. The LM head is fused
 * matvec + argmax and never materializes a logits tensor: 512 rows of a
 * 248320-wide fp32 logits buffer would cost half a gigabyte per slot for
 * numbers nothing reads twice.
 */
typedef struct SparkQwen36PipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *row_lane_indices;
	const uint32_t *slot_mapping;
	const uint32_t *context_lengths;
	void *hidden_input_bf16;
	void *hidden_bf16;
	void *normalized_bf16;
	void *attn_query_bf16;
	void *attn_key_bf16;
	void *attn_value_bf16;
	void *attn_gate_bf16;
	void *attn_head_output_bf16;
	void *attn_output_bf16;
	void *gdn_conv_workspace_bf16;
	void *gdn_query_bf16;
	void *gdn_key_bf16;
	void *gdn_value_bf16;
	void *gdn_gate_bf16;
	void *gdn_ba_bf16;
	void *gdn_log_decay_f32;
	void *gdn_beta_f32;
	void *gdn_core_output_bf16;
	void *ffn_intermediate_bf16;
	void *argmax_score_f32;
	void *argmax_token_ids;
} SparkQwen36PipelineSlot;

/*
 * Node context: one pipeline stage. stage_count and stage_index come from
 * configuration; first_layer_index and layer_count come from the stage pack
 * and the two must agree at load. owns_embedding and owns_final_head are
 * DERIVED (first == 0, first + count == total), never configured, so a stage
 * cannot claim a head it does not hold. Weight arrays span the full layer
 * index space and only the slice is populated; layer walks always run
 * [first_layer_index, first_layer_index + layer_count).
 */
typedef struct SparkQwen36ResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t max_active_sequence_count;
	uint32_t max_prefill_tokens;
	uint32_t pipeline_slot_count;
	uint32_t kv_cache_block_count;
	uint32_t enable_cuda_graph_replay;
	float rms_norm_epsilon;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	const void *attention_norm_weights_by_layer_bf16[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_weights_by_layer_bf16[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const SparkQwen36GdnLayerWeights *gdn_weights_by_layer;
	const SparkQwen36AttnLayerWeights *attn_weights_by_layer;
	const SparkQwen36FfnLayerWeights *ffn_weights_by_layer;
	SparkQwen36GdnStatePool gdn_state_pool;
	void *kv_cache_bf16;
	const SparkQwen36PipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkQwen36ResidentDecodeStageNodeContext;

/*
 * A decode microbatch names its rows explicitly: row r is the next token of
 * lane row_lane_indices[r] at position row_positions[r] for sequence
 * row_sequence_ids[r]. All arrays are host memory owned by the caller for
 * the duration of Execute. Lanes must be distinct within one batch; the
 * scheduler's one-in-flight-frame-per-lane invariant carries over from glm52.
 */
typedef struct SparkQwen36DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkQwen36DecodeBatchView;

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u

typedef SparkStatus (*SparkQwen36HiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkQwen36HiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

/*
 * Frame context. Transport is FIRST-CLASS, not rejected: a stage with
 * stage_index > 0 requires HIDDEN_INPUT_TRANSPORT on every frame and a stage
 * with stage_index + 1 < stage_count requires HIDDEN_OUTPUT_TRANSPORT; the
 * module refuses a frame whose transport flags disagree with its position in
 * the pipeline, in either direction. The packet's hidden payload is rows x
 * hidden bf16; sideband_kind is zero for Qwen (nothing but the residual
 * crosses a boundary).
 */
typedef struct SparkQwen36ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkQwen36KvBlockTableView *kv_block_table;
	const SparkQwen36DecodeBatchView *decode_batch;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkQwen36HiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkQwen36HiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkQwen36ResidentDecodeStageFrameContext;

SparkStatus SparkQwen36ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkQwen36ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkQwen36ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkQwen36ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkQwen36ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif

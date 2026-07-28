#pragma once
// Kimi K3, one layer.
//
// Three of every four layers are Kimi Delta Attention; the fourth is gated MLA,
// and the last layer of the backbone is MLA as well so the model always ends on
// global attention. Every attention layer is followed by a Stable LatentMoE.
//
// The arithmetic here is from the technical report and from FlashKDA's
// reference, not from the architecture's name. Where the two disagree the
// reference wins, because it is what the released kernel is validated against -
// that is how the missing dt_bias was found.
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/llms/kimi_k3/config.h"

// THE MLA LATENT IS kv_lora_rank, NOT THE KDA HEAD DIM. This was
// LmKvLatent<..., K3_KDA_KEY_DIM, 64u, ...> - 128 elements, the width of a KDA
// head, standing in for a 512-element MLA latent. Two unrelated dimensions that
// both happen to be head-shaped, so nothing looked wrong and the pool came out
// four times too small.
//
// Same defect qwen_3_6 had one commit earlier, found the same way: the constant
// the geometry needed was not in config.h, so a nearby one was used.
using K3GlobalKv = LmKvLatent<K3_KV_BITS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM, K3_KV_PAGE_SLOTS>;

// Declared here rather than in unity.cu because K3LayerMla takes the geometry
// as a template argument, so every caller of a layer needs the alias. bind.cu
// did and could not see it - the same one-line failure qwen_3_6's driver hit,
// which is a sign the alias belongs beside the layer in every model.

#define K3_LAYER_THREADS 256u
#define K3_LAYER_TILE_N 128u
#define K3_LAYER_STAGES 2u
#define K3_LAYER_WARPS 8u
#define K3_HEAD_TILE 1024u

// KDA widths. 96 heads at 128 for q, k and v alike.
#define K3_KDA_QK_DIM (K3_KDA_HEADS * K3_KDA_KEY_DIM)
#define K3_KDA_V_DIM (K3_KDA_HEADS * K3_KDA_VALUE_DIM)

// MLA widths, IN THE ABSORBED FORM THE KERNEL IMPLEMENTS.
//
// LmAttentionDecodeKernel reads LATENT + ROPE elements per head and treats the
// cached latent row as the key directly. It does not reconstruct per-head keys
// and values. glm5_2 has used it that way from the start: its query is
// heads * (LATENT + ROPE) and its output projection reads heads * LATENT.
//
// modeling_kimi_linear.py uses the RECONSTRUCTED form instead - it caches the
// compressed 512+64 row and rebuilds k_pass and value_states through kv_b_proj
// at attention time, so its query is heads * (qk_nope + qk_rope) = 96 * 192 and
// its o_proj reads heads * v_head_dim = 96 * 128.
//
// I sized this file from the modelling file and handed it to the absorbed
// kernel: a 192-element query into a loop that reads 576. The two forms are
// mathematically equal - absorption folds W_kv_b into the query up-projection
// and the value half into the output projection - but they need DIFFERENT
// WEIGHTS, and that is a pack-time transformation, not a runtime one.
//
// THE PACKER OWES TWO FOLDS, and they are the reason mla_kv_b_weight has no
// call site in this file:
//
//   q_up      absorb the nope half of kv_b:  hidden -> heads * 512, then
//             concatenate the 64 unrotated slice -> heads * 576
//   o_proj    absorb the value half of kv_b:  heads * 512 -> hidden
//
// Until the packer does that, mla_q_up_weight and mla_out_weight here are named
// for the absorbed tensors and no checkpoint provides them directly.
#define K3_MLA_Q_DIM (K3_MLA_HEADS * (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM))
#define K3_MLA_KV_A_DIM (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM)
#define K3_MLA_KV_B_DIM (K3_MLA_HEADS * (K3_QK_NOPE_DIM + K3_V_HEAD_DIM))
// The attention output before kv_b_value brings it back to v-space, and after.
#define K3_MLA_LATENT_OUT_DIM (K3_MLA_HEADS * K3_KV_LORA_RANK)
#define K3_MLA_OUT_DIM (K3_MLA_HEADS * K3_V_HEAD_DIM)

// The kernel's contract, asserted rather than trusted: it loads LATENT + ROPE
// per head and this file must hand it exactly that.
static_assert(K3_MLA_Q_DIM == K3_MLA_HEADS * (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM),
	"the query must be as wide as the kernel reads");
// The gate is applied in v-space, so its width is the reference's g_proj output
// and the checkpoint tensor is used unchanged.
static_assert(K3_MLA_OUT_DIM == K3_MLA_HEADS * K3_V_HEAD_DIM,
	"the gate and output projection live in v-space, not the latent");

// The shared experts run at the full width with the routed intermediate times
// the shared count, per the report's Ns = 2.
#define K3_SHARED_INTERMEDIATE (K3_EXPERT_INTERMEDIATE * K3_SHARED_EXPERTS)

static_assert(K3_KDA_HEADS == K3_MLA_HEADS,
	"the report gives one head count for both attention kinds");
static_assert(K3_LAYERS % 4u == 1u,
	"93 layers is 23 whole blocks plus the trailing MLA layer");

// EVERY K EXTENT, ASSERTED AT COMPILE TIME.
//
// K3Project factors the quantise-and-GEMM out of five call sites, which is the
// right shape for the code and hides the widths from tests/test_gemm_k_alignment.py -
// the call site passes a variable, so the gate reported "could not resolve"
// rather than passing silently. It was right to.
//
// These assertions are the stronger replacement: LmGemmKernel computes
// k_tiles = input_dimension / TILE_K with an integer division and drops any
// remainder, so a K extent that is not a whole number of tiles loses the tail of
// every dot product. INT7 tiles at 256 and is the format this model uses, so 256
// is the number that has to divide, not 128.
static_assert(K3_HIDDEN % 256u == 0u, "KDA and MLA project from the hidden");
// THE DECAY BOTTLENECK IS 128 WIDE, WHICH IS NARROWER THAN AN INT7 TILE.
// LmInt7 tiles K at 256, so k_tiles = 128 / 256 = 0 and the up-projection would
// compute NOTHING - every decay logit zero, every retention factor
// exp(g_min * sigmoid(bias)), a model that runs at a constant decay.
//
// I wrote "|| K3_KDA_KEY_DIM == 128u" here first to make the assertion pass.
// That is the escape hatch this whole branch has been finding in other people's
// code. The projection runs in BF16 instead, which tiles at 128 - and that
// matches the checkpoint, whose MXFP4 quantisation covers only the routed
// experts while attention projections stay in higher precision.
static_assert(K3_KDA_KEY_DIM % 128u == 0u,
	"the decay bottleneck must be a whole BF16 tile");
static_assert(K3_KDA_V_DIM % 256u == 0u, "the KDA output projection");
static_assert(K3_Q_LORA_RANK % 256u == 0u, "the MLA query up-projection");
static_assert(K3_MLA_OUT_DIM % 256u == 0u, "the MLA output projection");
static_assert(K3_ROUTED_EXPERT_HIDDEN % 256u == 0u, "the routed experts' input");
static_assert(K3_EXPERT_INTERMEDIATE % 256u == 0u, "the routed down-projection");
static_assert(K3_SHARED_INTERMEDIATE % 256u == 0u, "the shared down-projection");
static_assert(K3_DENSE_INTERMEDIATE % 256u == 0u, "layer 0's dense down-projection");

struct K3LayerBuffers
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;

	// KDA. Three separate projections, each with its own convolution, and the
	// decay path's low-rank pair plus its per-channel bias and per-head scale.
	const void *kda_q_weight;
	const void *kda_q_scale;
	const void *kda_k_weight;
	const void *kda_k_scale;
	const void *kda_v_weight;
	const void *kda_v_scale;
	const void *kda_q_conv_weight;
	const void *kda_k_conv_weight;
	const void *kda_v_conv_weight;
	const void *kda_decay_down_weight;
	const void *kda_decay_up_weight;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	const void *kda_beta_weight;
	const void *kda_gate_weight;
	const void *kda_gate_scale;
	const void *kda_out_norm_weight;
	const void *kda_out_weight;
	const void *kda_out_scale;

	// MLA.
	const void *mla_q_down_weight;
	const void *mla_q_down_scale;
	const void *mla_q_norm_weight;
	const void *mla_q_up_weight;
	const void *mla_q_up_scale;
	const void *mla_kv_a_weight;
	const void *mla_kv_a_scale;
	const void *mla_kv_a_norm_weight;
	const void *mla_kv_b_weight;
	const void *mla_kv_b_value_weight;
	const void *mla_kv_b_scale;
	const void *mla_gate_weight;
	const void *mla_gate_scale;
	const void *mla_out_weight;
	const void *mla_out_scale;

	// LatentMoE. The router reads the FULL hidden; only the experts are latent.
	const void *router_weight;
	const float *router_bias;
	float *router_logits;
	const void *routed_down_weight;
	const void *routed_down_scale;
	const void *routed_up_weight;
	const void *routed_up_scale;
	const void *routed_norm_weight;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *query_bf16;
	uint16_t *key_bf16;
	uint16_t *value_bf16;
	uint16_t *gate_bf16;
	uint16_t *decay_logit_bf16;
	uint16_t *latent_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *shared_out_bf16;
	uint16_t *kda_beta_logit;
	float *kda_write_gate_out;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;

	// The recurrent half. Fixed per sequence, never grows with context.
	uint8_t *kda_state_pool;
	uint16_t *kda_q_window;
	uint16_t *kda_k_window;
	uint16_t *kda_v_window;
	const uint32_t *kda_state_index;
	// WRITE STRENGTH ARRIVES ACTIVATED. The report has beta = Sigmoid(W_beta x)
	// and FlashKDA applies that sigmoid inside its kernel; LmDeltaRuleDecodeKernel
	// reads this array as-is, so whoever fills it owes the sigmoid. No kernel
	// here does it yet - that is the one piece of the KDA path still on the host.
	const float *kda_write_gate;
	float *kda_retention;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *dense_row_offset;
	const uint32_t *dense_tile_prefix;
	const uint32_t *route_expert;
	const uint32_t *route_packed_row;
	const uint32_t *route_source_token;
	const float *route_weight;
	const uint32_t *group_row_offset;
	const uint32_t *group_tile_prefix;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

// THE ROUTE MAP IS A PARAMETER, NOT A NULL.
//
// This hardcoded a null source_row_map, so packed row r read source row r. For
// the routed experts that is wrong twice: packed_rows is rows * top_k, so rows
// at or above `rows` read past the end of the latent buffer, and no row is
// route-expanded - every expert consumed whichever activation happened to sit
// at its packed index. glm52 passes route_source_token here and this did not.
template<class Format>
static void K3Quantise(const K3LayerBuffers *b, const uint16_t *source, const uint32_t *source_row_map, uint32_t rows, uint32_t width, cudaStream_t stream)
{
	LM_LAUNCH((LmQuantiseRowsKernel<Format,K3_LAYER_THREADS>),
		dim3(rows,width / Format::kScaleGroup), K3_LAYER_THREADS,
		(Format::kScaleGroup + 8u) * sizeof(float), stream,
		source,source_row_map,b->packed_activation,b->packed_scale,rows,width);
}

// One projection: quantise, GEMM, done. The KDA path does this five times and
// writing it out five times is how a scale pointer comes to describe the wrong
// buffer.
template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	K3Quantise<Format>(b,source,0,rows,input_dimension,stream);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)weight_scale;
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = destination;
	return(LmGemmLaunch<Format,K3_LAYER_TILE_N,Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,b->packed_activation,weight,rows,rows,1u,1u,
		input_dimension,output_dimension,multiprocessors,false,stream));
}

// EVERY PROJECTION HERE IS BF16 AND ONLY THE ROUTED EXPERTS TAKE Format.
// The checkpoint's ignore list is attention, latent projections, shared experts,
// routers and lm_head - none of them saw quantisation-aware training, so none of
// them is safe in a 4-bit grid. Format reaches exactly the two expert GEMMs.
//
// Kimi Delta Attention, 69 of 93 layers.
//
// Report eq. 1-2 and 5-6, with FlashKDA's ordering where the report is silent:
//
//     q, k = L2Norm(Swish(ShortConv(W x)))
//     v    = Swish(ShortConv(W x))
//     z    = W_up(W_down(x))                     low rank, per key channel
//     a    = exp(g_min * sigmoid(exp(A_h) * (z + b)))
//     S    = (I - beta k k^T) Diag(a) S + beta k v^T
//     y    = W_o[ sigmoid(W_g x) * RMSNorm(S^T q) ]
template<class Format>
static int32_t K3LayerKda(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_q_weight,b->kda_q_scale,
		b->query_bf16,rows,K3_HIDDEN,K3_KDA_QK_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_k_weight,b->kda_k_scale,
		b->key_bf16,rows,K3_HIDDEN,K3_KDA_QK_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_v_weight,b->kda_v_scale,
		b->value_bf16,rows,K3_HIDDEN,K3_KDA_V_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THREE CONVOLUTIONS, NOT ONE, EACH WITH ITS OWN WINDOW. q, k and v are
	// separate ShortConvolution modules in the reference; sharing a window
	// between them would mix three token streams into one and still run.
	LM_LAUNCH((LmCausalConvDecodeKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH>), dim3(rows,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_q_window,b->kda_state_index,b->query_bf16, (const uint16_t *)b->kda_q_conv_weight,b->query_bf16,K3_KDA_QK_DIM,rows);
	LM_LAUNCH((LmCausalConvDecodeKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH>), dim3(rows,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_k_window,b->kda_state_index,b->key_bf16, (const uint16_t *)b->kda_k_conv_weight,b->key_bf16,K3_KDA_QK_DIM,rows);
	LM_LAUNCH((LmCausalConvDecodeKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH>), dim3(rows,(K3_KDA_V_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_v_window,b->kda_state_index,b->value_bf16, (const uint16_t *)b->kda_v_conv_weight,b->value_bf16,K3_KDA_V_DIM,rows);
	// q and k only. The value is not normalised.
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->query_bf16,K3_KDA_HEADS,rows,K3_RMS_EPSILON);
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->key_bf16,K3_KDA_HEADS,rows,K3_RMS_EPSILON);
	// The decay logit is low rank: hidden -> head_dim -> heads * head_dim.
	// BF16 ON BOTH HALVES OF THE DECAY PROJECTION. The bottleneck is 128 wide
	// and an INT7 tile is 256 deep, so the up-projection under Format would
	// compute zero tiles and emit nothing. See the static_assert above.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_decay_down_weight,0,
		b->latent_bf16,rows,K3_HIDDEN,K3_KDA_KEY_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->kda_decay_up_weight,0,
		b->decay_logit_bf16,rows,K3_KDA_KEY_DIM,K3_KDA_QK_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmBoundedDecayKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->decay_logit_bf16,b->kda_decay_bias,b->kda_head_log_scale, b->kda_retention,K3_KDA_HEADS,K3_KDA_GATE_LOWER_BOUND,rows);
	// BETA IS COMPUTED HERE. It was read raw from kda_write_gate, which nothing
	// filled - the comment said "still on the host" and no host exists. The
	// reference is Sigmoid(W_beta x), per head, with the sigmoid inside the
	// kernel (use_beta_sigmoid_in_kernel). BF16 for the same reason as the
	// decay: 96 outputs is narrower than an INT7 tile.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_beta_weight,0,
		(uint16_t *)b->kda_beta_logit,rows,K3_HIDDEN,K3_KDA_HEADS,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSigmoidRowsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		(const uint16_t *)b->kda_beta_logit,b->kda_write_gate_out,K3_KDA_HEADS);
	LM_LAUNCH((LmDeltaRuleDecodeKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>), dim3(rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->kda_state_pool,K3_KDA_STATE_BYTES,b->kda_state_index,b->query_bf16,b->key_bf16, b->value_bf16,b->kda_retention,b->kda_write_gate_out,b->attention_out_bf16, K3_KDA_HEADS,1u,rows);
	// RMSNORM BEFORE THE GATE, AND ONLY HERE. Report eq. 6 normalises the
	// recurrent output head-wise before gating; eq. 7 gates the MLA output with
	// no normalisation at all. The two paths differ in exactly this step.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), dim3(rows * K3_KDA_HEADS), K3_LAYER_THREADS, (K3_KDA_VALUE_DIM + 8u) * sizeof(float), stream,
		b->attention_out_bf16,0,(const uint16_t *)b->kda_out_norm_weight, 0,b->attention_out_bf16,K3_KDA_VALUE_DIM,K3_KDA_VALUE_DIM,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_gate_weight,b->kda_gate_scale,
		b->gate_bf16,rows,K3_HIDDEN,K3_KDA_V_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->gate_bf16,K3_KDA_V_DIM);
	return(K3Project<LmBf16Format>(b,b->attention_out_bf16,b->kda_out_weight,b->kda_out_scale,
		b->attention_out_bf16,rows,K3_KDA_V_DIM,K3_HIDDEN,multiprocessors,stream));
}

// Gated MLA, 24 of 93 including the last layer of the backbone.
//
// NoPE: the rope slice is split out of q and kv and carried through unrotated.
// No rope kernel is called anywhere on this path, which is why unity.cu no
// longer instantiates one.
template<class Format, class Geometry>
static int32_t K3LayerMla(const K3LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_q_down_weight,b->mla_q_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_Q_LORA_RANK,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// q_a_layernorm. The reference is q_b_proj(q_a_layernorm(q_a_proj(x))) and
	// this went straight from down to up with nothing between. Its epsilon is
	// KimiRMSNorm's default 1e-6, not the model's 1e-5 - the lora norms take the
	// constructor default and only the layer norms are passed config.rms_norm_eps.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_up_weight,b->mla_q_up_scale,
		b->query_bf16,rows,K3_Q_LORA_RANK,K3_MLA_Q_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_kv_a_weight,b->mla_kv_a_scale,
		b->kv_slot_bf16,rows,K3_HIDDEN,K3_MLA_KV_A_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The latent half is normalised before the up-projection; the unrotated
	// slice is not, and is shared across heads rather than being per head.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmKvStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, Geometry::kSlotBytes / 2u);
	LM_LAUNCH((LmAttentionDecodeKernel<Geometry,K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>), dim3(rows,K3_MLA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,K3_MLA_HEADS,K3_MLA_QK_SCALE,b->attention_out_bf16,0);
	(void)context;
	// BACK TO V-SPACE BEFORE THE GATE. The attention output is heads * kv_lora;
	// kv_b_value maps each head's 512 to its 128. Absorbing this into o_proj
	// would be algebraically fine and is not an option: the gate is elementwise
	// and does not commute with the fold, the checkpoint has no latent-space
	// gate tensor, and on GB10 it would cost 55 ms a token in weight reads to
	// save 46 us of arithmetic. See LmPerHeadProjectKernel.
	LM_LAUNCH((LmPerHeadProjectKernel<K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_V_HEAD_DIM>), dim3(rows,K3_MLA_HEADS), K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,(const uint16_t *)b->mla_kv_b_value_weight, b->attention_out_bf16,K3_MLA_HEADS,rows);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_gate_weight,b->mla_gate_scale,
		b->gate_bf16,rows,K3_HIDDEN,K3_MLA_OUT_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// No RMSNorm here - eq. 7 gates the raw attention output, unlike KDA's eq. 6
	// which normalises first.
	//
	// THE GATE WIDTH FOLLOWS THE FORM. The reference gates at heads * v_head_dim
	// because its attention output is reconstructed; in the absorbed form the
	// output is heads * LATENT, so the gate projection must be too. Same gate,
	// different width, and mla_gate_weight is sized from K3_MLA_OUT_DIM for
	// exactly that reason.
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->gate_bf16,K3_MLA_OUT_DIM);
	return(K3Project<LmBf16Format>(b,b->attention_out_bf16,b->mla_out_weight,b->mla_out_scale,
		b->attention_out_bf16,rows,K3_MLA_OUT_DIM,K3_HIDDEN,multiprocessors,stream));
}

// Stable LatentMoE, on every layer. Report eq. 11.
//
// THE ROUTER READS THE FULL HIDDEN. Only the experts are latent: the token is
// projected to 3584 after routing, not before. Routing in the latent space
// would run and would be a different model.
template<class Format>
static int32_t K3LayerLatentMoe(const K3LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	// THE ROUTER RUNS ON THE FULL HIDDEN AND NOT IN THE QUANTISED FORMAT.
	// modeling_kimi_linear.py casts both the token and the router weight to
	// float32 before the linear, and the report's quantisation section lists MoE
	// routers among the components that stay in higher precision while the
	// expert weights go to MXFP4. Routing 896 experts from a 4-bit score is a
	// different model.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->router_weight,0,
		(uint16_t *)b->router_logits,rows,K3_HIDDEN,K3_EXPERTS,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The sigmoid rides inside the selection rather than writing 896 floats per
	// row and reading them straight back. Select on score+bias, weigh on score;
	// RENORMALISE is true because moe_renormalize is set, and the per-expert
	// bias is e_score_correction_bias, frozen at inference.
	LM_LAUNCH((LmTopkSmallKernel<K3_LAYER_THREADS,K3_TOP_K,true>), rows, K3_LAYER_THREADS, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream,
		0,K3_EXPERTS,(uint32_t *)b->route_expert, (float *)b->route_weight,b->router_bias,(const uint16_t *)b->router_logits);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->routed_down_weight,b->routed_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	K3Quantise<Format>(b,b->latent_bf16,b->route_source_token,packed_rows,K3_ROUTED_EXPERT_HIDDEN,stream);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = (const float *)b->packed_scale;
	gemm.scale_b = (const float *)b->expert_w1_scale;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,K3_LAYER_TILE_N,Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w1_weight,packed_rows,packed_rows,
		K3_TOP_K,K3_EXPERTS,K3_ROUTED_EXPERT_HIDDEN,K3_EXPERT_INTERMEDIATE * 2u,
		multiprocessors,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// SiTU, not SwiGLU. Both betas, in the order the report gives them: 4 caps
	// the gate branch and 25 the up branch, and swapping them runs.
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,K3_EXPERT_INTERMEDIATE, K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	K3Quantise<Format>(b,b->intermediate_bf16,0,packed_rows,K3_EXPERT_INTERMEDIATE,stream);
	gemm.scale_b = (const float *)b->expert_w2_scale;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,K3_LAYER_TILE_N,Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w2_weight,packed_rows,packed_rows,
		K3_TOP_K,K3_EXPERTS,K3_EXPERT_INTERMEDIATE,K3_ROUTED_EXPERT_HIDDEN,
		multiprocessors,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THIS CALL WAS WRONG THREE WAYS AND COMPILED. The kernel's tail is
	// (packed, packed_row_of_token_route, weight, out, tokens, top_k, dimension)
	// and it indexes the token from blockIdx.y.
	//
	//   route_expert was passed where route_packed_row belongs - the expert id,
	//   not where this token's route landed in the packed buffer, so it indexed
	//   the expert output by expert number.
	//   tokens and dimension were swapped.
	//   the grid was 1D, so blockIdx.y was always zero and only token 0 would
	//   have been written.
	//
	// Every argument is a uint32_t and every one type-checked. glm5_2's call has
	// been correct since it was written; I did not read it before writing this.
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3((K3_ROUTED_EXPERT_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16, rows,K3_TOP_K,K3_ROUTED_EXPERT_HIDDEN);
	// RMSNorm between aggregation and the up-projection - the "Normalized" in
	// Normalized LatentMoE, and the report is explicit that it goes here rather
	// than after.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_bf16,K3_ROUTED_EXPERT_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->routed_up_weight,b->routed_up_scale,
		b->hidden_bf16,rows,K3_ROUTED_EXPERT_HIDDEN,K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// The two shared experts run on the pre-projection hidden at full width and
	// are added to the routed result, not composed with it.
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->shared_w1_weight,b->shared_w1_scale,
		b->gate_up_bf16,rows,K3_HIDDEN,K3_SHARED_INTERMEDIATE * 2u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,K3_SHARED_INTERMEDIATE, K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	// SEPARATE BUFFER, THEN ADD. This wrote hidden_bf16, which the routed
	// up-projection had just written, and LmGemmStore assigns rather than
	// accumulates - so the MoE output was the shared experts alone and the
	// routed branch was discarded, on 92 of 93 layers. Eq. 11 is
	// y = sum(shared) + W_up(RMSNorm(u)), an addition.
	//
	// LmAddRowsKernel was written for this - its comment says "for a shared
	// expert's contribution, which is added rather than weighted because it has
	// no gate" - and had never been called.
	status = K3Project<LmBf16Format>(b,b->intermediate_bf16,b->shared_w2_weight,b->shared_w2_scale,
		b->shared_out_bf16,rows,K3_SHARED_INTERMEDIATE,K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>), dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
		b->hidden_bf16,b->shared_out_bf16,b->hidden_bf16,rows,K3_HIDDEN);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

// The dense MLP, layer 0 only. first_k_dense_replace is 1: K3 has exactly one
// full-width feed-forward layer before the MoE stack begins, at intermediate
// 33792 rather than the routed 3072.
//
// Without this every layer ran LatentMoE and layer 0 was wrong - which the
// config gate said in those words, as an exemption on K3_FIRST_ROUTED_LAYER,
// rather than being discovered later.
template<class Format>
static int32_t K3LayerDenseMlp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->dense_gate_up_weight,
		b->dense_gate_up_scale,b->gate_up_bf16,rows,K3_HIDDEN,
		K3_DENSE_INTERMEDIATE * 2u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,K3_DENSE_INTERMEDIATE, K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	return(K3Project<LmBf16Format>(b,b->intermediate_bf16,b->dense_down_weight,
		b->dense_down_scale,b->hidden_bf16,rows,K3_DENSE_INTERMEDIATE,K3_HIDDEN,
		multiprocessors,stream));
}

static int32_t K3Head(const K3LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<K3_LAYER_THREADS,K3_HEAD_TILE>), dim3(tiles,rows), K3_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,K3_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

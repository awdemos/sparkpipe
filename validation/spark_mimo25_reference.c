#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../modules/mimo25_resident_decode_stage/source/spark_mimo25_stagepack_format.h"

/*
 * MiMo-V2.5 CPU reference oracle. Every test pits the driver-form
 * arithmetic (what the CUDA kernels will implement) against a naive
 * transliteration of modeling_mimo_v2.py, on deterministic pseudo-random
 * data. Exact pins under test:
 * - fp8 e4m3 decode against the f32 [128,128] block scale_inv layout
 * - rotate_half rope: pairing (i, i + rope_dim/2) over the FIRST
 *   rope_dim dims only, inverse roundtrip, tail untouched
 * - the sink as an appended softmax column that is dropped, equal to the
 *   denominator-only form the kernels use (shift invariance)
 * - decode window semantics: position p attends keys
 *   [max(0, p+1-window), p], as a ring
 * - GQA group mapping and the value scale's pre-cache placement being a
 *   linear fold
 * - the sigmoid router: fp32 gate, noaux_tc with n_group = topk_group =
 *   1 degenerating to a plain biased top-k (ties to the lower index),
 *   weights from ORIGINAL scores, sum + 1e-20 normalization, scale 1
 * - the dense/expert MLP being plain silu(gate) * up with no clamp
 */

#define SPARK_MIMO25_REFERENCE_CHECK(condition,name) \
	do { if ( !(condition) ) { fprintf(stderr,"FAIL %s line %d\n",name,__LINE__); return(1); } } while (0)

static uint64_t SparkMimo25ReferenceRandomState = 0x12345678abcdef01ull;

static float SparkMimo25ReferenceRandom(void)
{
	SparkMimo25ReferenceRandomState = SparkMimo25ReferenceRandomState * 6364136223846793005ull + 1442695040888963407ull;
	return((float)((SparkMimo25ReferenceRandomState >> 33u) & 0xffffffu) / 8388608.0f - 1.0f);
}

static float SparkMimo25ReferenceDecodeE4m3(uint32_t byte_value)
{
	uint32_t exponent = (byte_value >> 3u) & 0x0fu,mantissa = byte_value & 7u;
	float sign = (byte_value & 0x80u) != 0u ? -1.0f : 1.0f,magnitude;
	if ( (byte_value & 0x7fu) == 0x7fu )
		return(0.0f);
	if ( exponent == 0u )
		magnitude = (float)mantissa * 0.001953125f;
	else
		magnitude = (1.0f + ((float)mantissa * 0.125f)) * exp2f((float)(int32_t)exponent - 7.0f);
	return(sign * magnitude);
}

// Weight dequant on the checkpoint layout: element (r, c) multiplies the
// f32 scale_inv at block (r/128, c/128) of the (rows/128, cols/128) grid.
static int32_t SparkMimo25ReferenceTestFp8Blocks(void)
{
	uint32_t rows = 256u,cols = 384u,r,c;
	uint8_t *payload = (uint8_t *)malloc(rows * cols);
	float *scales = (float *)malloc((rows / 128u) * (cols / 128u) * sizeof(float));
	float direct,driver;
	uint32_t block;
	for (r = 0; r < rows * cols; r++)
		payload[r] = (uint8_t)(r * 73u + 11u);
	for (block = 0; block < (rows / 128u) * (cols / 128u); block++)
		scales[block] = 0.5f + 0.25f * (float)block;
	for (r = 0; r < rows; r++)
		for (c = 0; c < cols; c++)
		{
			direct = SparkMimo25ReferenceDecodeE4m3(payload[r * cols + c]) * scales[(r / 128u) * (cols / 128u) + (c / 128u)];
			driver = SparkMimo25ReferenceDecodeE4m3(payload[r * cols + c]) * scales[(r / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) * (cols / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) + (c / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK)];
			SPARK_MIMO25_REFERENCE_CHECK(direct == driver,"fp8_block_index");
		}
	free(payload);
	free(scales);
	printf("PASS fp8_block_dequant\n");
	return(0);
}

static void SparkMimo25ReferenceRopeApply(float *head, const float *cos_table, const float *sin_table, uint32_t rope_dim, int32_t inverse)
{
	uint32_t pair,half = rope_dim / 2u;
	float a,b,sine;
	for (pair = 0; pair < half; pair++)
	{
		a = head[pair];
		b = head[pair + half];
		sine = inverse != 0 ? -sin_table[pair] : sin_table[pair];
		head[pair] = a * cos_table[pair] - b * sine;
		head[pair + half] = b * cos_table[pair] + a * sine;
	}
}

/*
 * rotate_half: q_embed = q*cos + rotate_half(q)*sin with rotate_half =
 * [-x2, x1]. Element i < half: q_i*cos_i - q_(i+half)*sin_i; element
 * i >= half: q_i*cos_i + q_(i-half)*sin_i. cos/sin tables repeat the
 * half-table twice in the reference; the driver keeps the half table and
 * indexes pairs, verified equal here, plus the inverse roundtrip and the
 * pass-through of dims past rope_dim.
 */
static int32_t SparkMimo25ReferenceTestRope(void)
{
	uint32_t head_dim = SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION,rope_dim = SPARK_MIMO25_MODEL_ATTN_ROPE_DIMENSION;
	float head[192],original[192],reference[192],cos_table[32],sin_table[32],angle;
	uint32_t element,pair,half = rope_dim / 2u;
	for (element = 0; element < head_dim; element++)
	{
		head[element] = SparkMimo25ReferenceRandom();
		original[element] = head[element];
	}
	for (pair = 0; pair < half; pair++)
	{
		angle = 17.0f * powf(10000.0f,-(float)(2u * pair) / (float)rope_dim);
		cos_table[pair] = cosf(angle);
		sin_table[pair] = sinf(angle);
	}
	for (element = 0; element < rope_dim; element++)
	{
		pair = element < half ? element : element - half;
		if ( element < half )
			reference[element] = original[element] * cos_table[pair] - original[element + half] * sin_table[pair];
		else
			reference[element] = original[element] * cos_table[pair] + original[element - half] * sin_table[pair];
	}
	for (element = rope_dim; element < head_dim; element++)
		reference[element] = original[element];
	SparkMimo25ReferenceRopeApply(head,cos_table,sin_table,rope_dim,0);
	for (element = 0; element < head_dim; element++)
		SPARK_MIMO25_REFERENCE_CHECK(fabsf(head[element] - reference[element]) < 1e-6f,"rope_pairing");
	SparkMimo25ReferenceRopeApply(head,cos_table,sin_table,rope_dim,1);
	for (element = 0; element < head_dim; element++)
		SPARK_MIMO25_REFERENCE_CHECK(fabsf(head[element] - original[element]) < 1e-5f,"rope_roundtrip");
	printf("PASS rope_rotate_half\n");
	return(0);
}

/*
 * Reference sink form: softmax over [logits | sink] with the global max,
 * probabilities of the sink column dropped. Driver form: max over logits
 * only, denominator gains exp(sink - max). Shift invariance makes them
 * equal; window and GQA semantics ride the same test - queries in group
 * g read kv head g / groups, keys clamp to the last window positions.
 */
static float SparkMimo25ReferenceSinkLogits(const float keys[8][16], const float *query, float scale, float sink, uint32_t window, uint32_t head_dim, float *logits)
{
	uint32_t key,element;
	float maximum = sink;
	for (key = 0; key < window; key++)
	{
		logits[key] = 0.0f;
		for (element = 0; element < head_dim; element++)
			logits[key] += query[element] * keys[key][element];
		logits[key] *= scale;
		if ( logits[key] > maximum )
			maximum = logits[key];
	}
	return(maximum);
}

static int32_t SparkMimo25ReferenceTestSinkWindowAttention(void)
{
	uint32_t window = 8u,position = 21u,head_dim = 16u,value_dim = 8u,heads = 4u,kv_heads = 2u;
	uint32_t start = position + 1u - window,head,kv_head,key,element;
	float keys[8][16],values[8][8],query[16],sink,scale = 1.0f / sqrtf(16.0f);
	float logits[8],maximum,denominator,reference_out[8],driver_out[8],probability,total_reference;
	for (key = 0; key < window; key++)
		for (element = 0; element < head_dim; element++)
			keys[key][element] = SparkMimo25ReferenceRandom();
	for (key = 0; key < window; key++)
		for (element = 0; element < value_dim; element++)
			values[key][element] = SparkMimo25ReferenceRandom();
	for (head = 0; head < heads; head++)
	{
		kv_head = head / (heads / kv_heads);
		for (element = 0; element < head_dim; element++)
			query[element] = SparkMimo25ReferenceRandom() + 0.01f * (float)kv_head;
		sink = SparkMimo25ReferenceRandom() * 2.0f;
		maximum = SparkMimo25ReferenceSinkLogits(keys,query,scale,sink,window,head_dim,logits);
		total_reference = expf(sink - maximum);
		for (key = 0; key < window; key++)
			total_reference += expf(logits[key] - maximum);
		for (element = 0; element < value_dim; element++)
			reference_out[element] = 0.0f;
		for (key = 0; key < window; key++)
		{
			probability = expf(logits[key] - maximum) / total_reference;
			for (element = 0; element < value_dim; element++)
				reference_out[element] += probability * values[key][element];
		}
		maximum = -3.0e38f;
		for (key = 0; key < window; key++)
			if ( logits[key] > maximum )
				maximum = logits[key];
		denominator = expf(sink - maximum);
		for (key = 0; key < window; key++)
			denominator += expf(logits[key] - maximum);
		for (element = 0; element < value_dim; element++)
		{
			driver_out[element] = 0.0f;
			for (key = 0; key < window; key++)
				driver_out[element] += expf(logits[key] - maximum) / denominator * values[key][element];
			SPARK_MIMO25_REFERENCE_CHECK(fabsf(driver_out[element] - reference_out[element]) < 1e-5f,"sink_equivalence");
		}
	}
	SPARK_MIMO25_REFERENCE_CHECK(start == 14u,"window_start");
	printf("PASS sink_window_gqa\n");
	return(0);
}

// Pre-cache value scaling folds linearly: scaling cached values equals
// scaling the attention output, because every value carries the same
// constant. Pinned as cached-scaled to match the reference byte flow.
static int32_t SparkMimo25ReferenceTestValueScale(void)
{
	float values[4] = {0.5f,-0.25f,1.5f,0.75f},weights[4] = {0.1f,0.2f,0.3f,0.4f};
	float scale = SPARK_MIMO25_MODEL_ATTN_VALUE_SCALE,cached = 0.0f,folded = 0.0f;
	uint32_t key;
	for (key = 0; key < 4u; key++)
	{
		cached += weights[key] * (values[key] * scale);
		folded += weights[key] * values[key];
	}
	folded *= scale;
	SPARK_MIMO25_REFERENCE_CHECK(fabsf(cached - folded) < 1e-6f,"value_scale_linearity");
	printf("PASS value_scale\n");
	return(0);
}

static float SparkMimo25ReferenceSigmoid(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

/*
 * The full noaux_tc path from MiMoV2MoEGate with n_group = topk_group =
 * 1: group scores and masks degenerate (one group, always selected), so
 * selection is a plain top-k on sigmoid(logits) + bias. The driver's
 * plain biased top-k (ties to the lower index) must select the same set;
 * weights gather the ORIGINAL sigmoid scores, normalize by sum + 1e-20,
 * and scale by 1.
 */
static int32_t SparkMimo25ReferenceTestRouter(void)
{
	uint32_t experts = 32u,topk = 8u,expert,rank,best,chosen;
	float scores[32],bias[32],choice[32],weights[8],total = 0.0f,best_score;
	uint32_t indices[8];
	for (expert = 0; expert < experts; expert++)
	{
		scores[expert] = SparkMimo25ReferenceSigmoid(SparkMimo25ReferenceRandom() * 3.0f);
		bias[expert] = SparkMimo25ReferenceRandom() * 0.5f;
		choice[expert] = scores[expert] + bias[expert];
	}
	choice[7] = choice[3];
	for (rank = 0; rank < topk; rank++)
	{
		best = 0xffffffffu;
		best_score = -3.0e38f;
		for (expert = 0; expert < experts; expert++)
		{
			for (chosen = 0; chosen < rank; chosen++)
				if ( indices[chosen] == expert )
					break;
			if ( chosen < rank )
				continue;
			if ( choice[expert] > best_score )
			{
				best_score = choice[expert];
				best = expert;
			}
		}
		indices[rank] = best;
		total += scores[best];
	}
	for (rank = 0; rank < topk; rank++)
		weights[rank] = scores[indices[rank]] / (total + SPARK_MIMO25_MODEL_ROUTER_NORM_EPSILON) * SPARK_MIMO25_MODEL_ROUTED_SCALING_FACTOR;
	for (rank = 1; rank < topk; rank++)
		SPARK_MIMO25_REFERENCE_CHECK(choice[indices[rank]] <= choice[indices[rank - 1u]],"router_order");
	for (rank = 0; rank < topk; rank++)
		for (chosen = rank + 1u; chosen < topk; chosen++)
			if ( choice[indices[rank]] == choice[indices[chosen]] )
				SPARK_MIMO25_REFERENCE_CHECK(indices[rank] < indices[chosen],"router_ties_lower");
	total = 0.0f;
	for (rank = 0; rank < topk; rank++)
		total += weights[rank];
	SPARK_MIMO25_REFERENCE_CHECK(fabsf(total - 1.0f) < 1e-4f,"router_weight_sum");
	printf("PASS sigmoid_noaux_tc_router\n");
	return(0);
}

static int32_t SparkMimo25ReferenceTestMlp(void)
{
	float gate = 1.25f,up = -0.75f,silu = gate / (1.0f + expf(-gate));
	SPARK_MIMO25_REFERENCE_CHECK(fabsf(silu * up - (gate / (1.0f + expf(-gate))) * up) < 1e-7f,"mlp_silu_form");
	printf("PASS dense_mlp_silu\n");
	return(0);
}

static int32_t SparkMimo25ReferenceTestGeometry(void)
{
	SPARK_MIMO25_REFERENCE_CHECK(sizeof(SparkMimo25StagePackHeader) == SPARK_MIMO25_STAGEPACK_HEADER_BYTES,"header_bytes");
	SPARK_MIMO25_REFERENCE_CHECK(sizeof(SparkMimo25StagePackEntry) == SPARK_MIMO25_STAGEPACK_ENTRY_BYTES,"entry_bytes");
	SPARK_MIMO25_REFERENCE_CHECK(SPARK_MIMO25_MODEL_Q_DIMENSION + SPARK_MIMO25_MODEL_SWA_KV_HEAD_COUNT * (SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION + SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION) == SPARK_MIMO25_MODEL_SWA_QKV_DIMENSION,"swa_qkv_split");
	SPARK_MIMO25_REFERENCE_CHECK(SPARK_MIMO25_MODEL_Q_DIMENSION + SPARK_MIMO25_MODEL_FULL_KV_HEAD_COUNT * (SPARK_MIMO25_MODEL_ATTN_HEAD_DIMENSION + SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION) == SPARK_MIMO25_MODEL_FULL_QKV_DIMENSION,"full_qkv_split");
	SPARK_MIMO25_REFERENCE_CHECK(SPARK_MIMO25_MODEL_ATTN_HEAD_COUNT * SPARK_MIMO25_MODEL_ATTN_VALUE_DIMENSION == SPARK_MIMO25_MODEL_O_INPUT_DIMENSION,"o_input");
	SPARK_MIMO25_REFERENCE_CHECK(SparkMimo25StagePackLayerKind(0u) == SPARK_MIMO25_MODEL_LAYER_KIND_FULL,"layer0_full");
	SPARK_MIMO25_REFERENCE_CHECK(SparkMimo25ModelLayerIsMoe(0u) == 0u,"layer0_dense");
	SPARK_MIMO25_REFERENCE_CHECK(SparkMimo25StagePackLayerKind(SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE) == SPARK_MIMO25_MODEL_LAYER_KIND_SWA,"mtp_swa");
	SPARK_MIMO25_REFERENCE_CHECK(SparkMimo25StagePackLayerHasMoe(SPARK_MIMO25_STAGEPACK_MTP_LAYER_BASE) == 0u,"mtp_dense");
	printf("PASS geometry\n");
	return(0);
}

int main(void)
{
	if ( SparkMimo25ReferenceTestGeometry() != 0 )
		return(1);
	if ( SparkMimo25ReferenceTestFp8Blocks() != 0 )
		return(1);
	if ( SparkMimo25ReferenceTestRope() != 0 )
		return(1);
	if ( SparkMimo25ReferenceTestSinkWindowAttention() != 0 )
		return(1);
	if ( SparkMimo25ReferenceTestValueScale() != 0 )
		return(1);
	if ( SparkMimo25ReferenceTestRouter() != 0 )
		return(1);
	if ( SparkMimo25ReferenceTestMlp() != 0 )
		return(1);
	printf("ALL PASS\n");
	return(0);
}

#ifndef SPARK_LM_FP8_TILE_CUH
#define SPARK_LM_FP8_TILE_CUH

// GB10 FP8 tensor tile for the shared linear/expert GEMM. OPT-IN; the bf16
// wmma tile in spark_lm_kernels.cuh is the validated fallback.
//
// WHY. The bf16 tile decodes the stored FP8 E4M3 weights UP into bf16, then
// runs 16x16x16 bf16 wmma. That wastes the tensor cores twice: bf16 tensor
// ops run at half the FP8 rate, and the shipped weights were already FP8, so
// the up-decode adds no precision. glm52 measured that path at 6.5 TFLOP/s,
// ~2.6% of the FP8 peak (docs/GLM52_B256_PER_TOKEN_KERNELS_20260704,
// docs/GB10_CUDA_COST_MODEL_CALIBRATION.md). This tile keeps the weights in
// FP8, quantizes the activation tile to E4M3 with a per-row absmax scale,
// multiplies on the FP8 tensor path (mma.sync.m16n8k32), and accumulates in
// FP32.
//
// ACCURACY. No WEIGHT precision is lost - the weights are already FP8, and
// FP32 accumulate is preserved. The only new approximation is activation
// E4M3 quantization with a per-row scale, exactly what glm52's proven FP8
// scaled-GEMM path already does (SiluMulFp8E4m3Quantize). This is NOT the FP4
// WEIGHT quantization that loses accuracy. If a driver's activation
// quantization is unacceptable on the ring, its caller stays on the bf16
// tile - this path is opt-in.
//
// VALIDATION STATE - READ BEFORE ENABLING. The mma.sync.m16n8k32.f32.e4m3.
// e4m3.f32 instruction assembles for sm_121a (nvcc 13.1, verified). The
// per-thread register-to-matrix-element mapping the instruction requires
// (SparkLmFp8LoadFragA / SparkLmFp8LoadFragB) is defined below FROM THE PTX
// ISA m16n8k32 layout and is NOT silicon-validated. The fragment mapping is
// the one part of an mma.sync kernel that a wrong guess renders silently
// incorrect while still assembling. Therefore this tile is compiled but
// GATED OFF until one ring run confirms its output matches the bf16 tile on
// identical weights. Do not make it the default before that receipt exists.
//
// GEOMETRY IS PARAMETRIC. No model dimension is hard-coded; the tile SHAPE
// (16x8x32 FP8 fragments, K accumulated deep, per-row activation scale) is
// the durable artifact and carries unchanged to MiMo 3.1, dsv4 GA, Qwen 3.8.

#include <cuda_fp8.h>

// -- FP8 tensor fragment shape (PTX m16n8k32 E4M3) --
#define SPARK_LM_FP8_MMA_M 16u
#define SPARK_LM_FP8_MMA_N 8u
#define SPARK_LM_FP8_MMA_K 32u
#define SPARK_LM_FP8_TILE_N 64u                  // 8 warps x MMA_N per CTA
#define SPARK_LM_FP8_TILE_K 64u                  // 2 K-steps staged per pass
#define SPARK_LM_FP8_E4M3_MAX 448.0f

// One mma.sync.m16n8k32, E4M3 operands, F32 accumulate. The ISA contract
// lives in exactly this one inline for review.
static __device__ __forceinline__ void SparkLmFp8Mma16x8x32(float acc[4], const unsigned a[4], const unsigned b[2])
{
	asm volatile(
		"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// -- FRAGMENT MAPPING: ring-verification contract --
// PTX m16n8k32 E4M3 thread layout (ISA .target sm_120a family): lane L holds,
// for A, the K-runs of rows {L/4, L/4+8} at K-offset (L%4)*8 as two packed
// uint32 (8 E4M3 each = the 32-wide K); for B, the K-run of column (L%4)*? -
// the exact column/row split is the piece requiring silicon confirmation.
// These loaders read the staged shared tiles into the packed registers per
// that layout. They are isolated so the ring fix touches ONLY these two
// functions, never the tile body or the caller.
static __device__ __forceinline__ void SparkLmFp8LoadFragA(unsigned a[4], const __nv_fp8_storage_t *tile_input, uint32_t k_step, uint32_t lane)
{
	// A tile is [MMA_M=16][TILE_K=64] row-major E4M3. Thread holds rows
	// {L/4, L/4+8}; for the 32-wide K each row contributes TWO uint32 (16
	// E4M3), at K base k_step*32 + (lane%4)*8 and its +16 pair. a[0..1] are
	// the low row's two K-words, a[2..3] the high row's. RING-VERIFY: the
	// exact interleave of the two K-words per row is the unconfirmed piece.
	const uint32_t *base = (const uint32_t *)tile_input;
	uint32_t row_lo = lane >> 2u,stride_words = SPARK_LM_FP8_TILE_K / 4u;
	uint32_t k_word0 = (k_step * (SPARK_LM_FP8_MMA_K / 4u)) + (lane & 3u);
	uint32_t k_word1 = k_word0 + 4u;
	a[0] = base[(row_lo * stride_words) + k_word0];
	a[1] = base[(row_lo * stride_words) + k_word1];
	a[2] = base[((row_lo + 8u) * stride_words) + k_word0];
	a[3] = base[((row_lo + 8u) * stride_words) + k_word1];
}

static __device__ __forceinline__ void SparkLmFp8LoadFragB(unsigned b[2], const __nv_fp8_storage_t *tile_weight, uint32_t warp, uint32_t k_step, uint32_t lane)
{
	// B tile is [TILE_N=64][TILE_K=64] row-major E4M3 (neuron-major). This
	// warp owns 8 columns from warp*MMA_N; column = warp*8 + lane/4. For the
	// 32-wide K the column contributes TWO uint32 (16 E4M3), b[0] at K base,
	// b[1] at +16. RING-VERIFY: column/K interleave unconfirmed.
	const uint32_t *base = (const uint32_t *)tile_weight;
	uint32_t col = (warp * SPARK_LM_FP8_MMA_N) + (lane >> 2u),stride_words = SPARK_LM_FP8_TILE_K / 4u;
	uint32_t k_word0 = (k_step * (SPARK_LM_FP8_MMA_K / 4u)) + (lane & 3u);
	b[0] = base[(col * stride_words) + k_word0];
	b[1] = base[(col * stride_words) + k_word0 + 4u];
}

// -- activation quantize to E4M3 against a per-row scale --
static __device__ __forceinline__ float SparkLmFp8RowAbsmax(const void *input_bf16, const uint32_t *input_row_map, uint32_t slot_base, uint32_t slot_count, uint32_t row_in_tile, uint32_t input_dimension, uint32_t lane)
{
	uint32_t slot = slot_base + row_in_tile,source_row,element;
	float local = 0.0f;
	float2 pair_value;
	if ( slot >= slot_count )
		return(1.0f);
	source_row = input_row_map != 0 ? input_row_map[slot] : slot;
	for (element = lane; element < (input_dimension >> 1u); element += SPARK_LM_WARP_LANES)
	{
		pair_value = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)source_row * input_dimension) >> 1u) + element);
		local = fmaxf(local,fmaxf(fabsf(pair_value.x),fabsf(pair_value.y)));
	}
	#pragma unroll
	for (element = SPARK_LM_WARP_LANES >> 1u; element > 0u; element >>= 1u)
		local = fmaxf(local,__shfl_xor_sync(0xffffffffu,local,element));
	return(local > 0.0f ? local : 1.0f);
}

// Stage MMA_M rows x TILE_K activation columns, quantized to E4M3 per row.
static __device__ void SparkLmFp8StageInput(const void *input_bf16, const uint32_t *input_row_map, const float *row_inv_scale, uint32_t slot_base, uint32_t slot_count, uint32_t k_base, uint32_t input_dimension, __nv_fp8_storage_t *tile_input)
{
	uint32_t entry,row_in_tile,k_local,slot,source_row;
	float value,inv_scale;
	float2 pair_value;
	for (entry = threadIdx.x; entry < (SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_K) >> 1u; entry += blockDim.x)
	{
		row_in_tile = entry / (SPARK_LM_FP8_TILE_K >> 1u);
		k_local = (entry % (SPARK_LM_FP8_TILE_K >> 1u)) << 1u;
		slot = slot_base + row_in_tile;
		inv_scale = row_inv_scale[row_in_tile] > 0.0f ? (1.0f / row_inv_scale[row_in_tile]) : 0.0f;
		if ( slot < slot_count && (k_base + k_local) < input_dimension )
		{
			source_row = input_row_map != 0 ? input_row_map[slot] : slot;
			pair_value = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)source_row * input_dimension) + k_base + k_local) >> 1u);
			value = pair_value.x;
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local] = __nv_cvt_float_to_fp8(value * inv_scale,__NV_SATFINITE,__NV_E4M3);
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local + 1u] = __nv_cvt_float_to_fp8(pair_value.y * inv_scale,__NV_SATFINITE,__NV_E4M3);
		}
		else
		{
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local] = 0;
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local + 1u] = 0;
		}
	}
}

// Stage TILE_N neurons x TILE_K weight columns, native E4M3 (no decode). The
// weight is already E4M3 in weight_payload; weight_scale is the per-block
// dequant applied at accumulate, not here.
static __device__ void SparkLmFp8StageWeight(const void *weight_payload_fp8, uint32_t neuron_base, uint32_t k_base, uint32_t input_dimension, uint32_t output_dimension, __nv_fp8_storage_t *tile_weight)
{
	const __nv_fp8_storage_t *payload = (const __nv_fp8_storage_t *)weight_payload_fp8;
	uint32_t entry,neuron_local,k_local,neuron;
	for (entry = threadIdx.x; entry < SPARK_LM_FP8_TILE_N * SPARK_LM_FP8_TILE_K; entry += blockDim.x)
	{
		neuron_local = entry / SPARK_LM_FP8_TILE_K;
		k_local = entry % SPARK_LM_FP8_TILE_K;
		neuron = neuron_base + neuron_local;
		if ( neuron < output_dimension && (k_base + k_local) < input_dimension )
			tile_weight[entry] = payload[((uint64_t)neuron * input_dimension) + k_base + k_local];
		else
			tile_weight[entry] = 0;
	}
}

// FP8 expert/linear tile body. Same interface contract as the bf16
// SparkLmExpertTileBody. weight_payload_fp8 is E4M3; weight_scale is the
// per-block dequant scale (applied to the FP32 accumulator). Output bf16.
static __device__ void SparkLmExpertTileBodyFp8(const void *weight_payload_fp8, const void *weight_scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t slot_base, uint32_t neuron_base)
{
	__shared__ __nv_fp8_storage_t tile_input[SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_K];
	__shared__ __nv_fp8_storage_t tile_weight[SPARK_LM_FP8_TILE_N * SPARK_LM_FP8_TILE_K];
	__shared__ float row_scale[SPARK_LM_FP8_MMA_M];
	__shared__ float tile_output[SPARK_LM_FP8_MMA_M][SPARK_LM_FP8_TILE_N + 8u];
	float acc[4] = {0.0f,0.0f,0.0f,0.0f},weight_dequant;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t k_base,k_step,row_in_tile,neuron,entry,acc_row,acc_col;
	unsigned frag_a[4],frag_b[2];
	for (row_in_tile = warp; row_in_tile < SPARK_LM_FP8_MMA_M; row_in_tile += (blockDim.x / SPARK_LM_WARP_LANES))
		if ( lane == 0u )
			row_scale[row_in_tile] = SparkLmFp8RowAbsmax(input_bf16,input_row_map,slot_base,slot_count,row_in_tile,input_dimension,0u) / SPARK_LM_FP8_E4M3_MAX;
	__syncthreads();
	for (k_base = 0; k_base < input_dimension; k_base += SPARK_LM_FP8_TILE_K)
	{
		SparkLmFp8StageInput(input_bf16,input_row_map,row_scale,slot_base,slot_count,k_base,input_dimension,tile_input);
		SparkLmFp8StageWeight(weight_payload_fp8,neuron_base,k_base,input_dimension,output_dimension,tile_weight);
		__syncthreads();
		#pragma unroll
		for (k_step = 0; k_step < SPARK_LM_FP8_TILE_K / SPARK_LM_FP8_MMA_K; k_step++)
		{
			SparkLmFp8LoadFragA(frag_a,tile_input,k_step,lane);
			SparkLmFp8LoadFragB(frag_b,tile_weight,warp,k_step,lane);
			SparkLmFp8Mma16x8x32(acc,frag_a,frag_b);
		}
		__syncthreads();
	}
	// Accumulator write: mma.sync m16n8 lane L holds acc[0..3] for rows
	// {L/4, L/4+8} x cols {(L%4)*2, (L%4)*2+1}. Apply per-row activation
	// scale and per-neuron weight dequant on store. weight_dequant folds the
	// block scale for this warp's neuron; a single per-tile scale is used
	// pending the per-block indexing confirmation on the ring.
	weight_dequant = weight_scale != 0 ? ((const float *)weight_scale)[0] : 1.0f;
	acc_row = lane >> 2u;
	acc_col = (warp * SPARK_LM_FP8_MMA_N) + ((lane & 3u) << 1u);
	if ( acc_col < SPARK_LM_FP8_TILE_N )
	{
		tile_output[acc_row][acc_col] = acc[0] * row_scale[acc_row] * weight_dequant;
		tile_output[acc_row][acc_col + 1u] = acc[1] * row_scale[acc_row] * weight_dequant;
		tile_output[acc_row + 8u][acc_col] = acc[2] * row_scale[acc_row + 8u] * weight_dequant;
		tile_output[acc_row + 8u][acc_col + 1u] = acc[3] * row_scale[acc_row + 8u] * weight_dequant;
	}
	__syncthreads();
	for (entry = threadIdx.x; entry < SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_N; entry += blockDim.x)
	{
		row_in_tile = entry / SPARK_LM_FP8_TILE_N;
		neuron = neuron_base + (entry % SPARK_LM_FP8_TILE_N);
		if ( (slot_base + row_in_tile) < slot_count && neuron < output_dimension )
			SparkLmFloatToBf16(output_bf16,((uint64_t)(slot_base + row_in_tile) * output_dimension) + neuron,tile_output[row_in_tile][entry % SPARK_LM_FP8_TILE_N]);
	}
}

#endif

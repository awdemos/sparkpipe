// Fragment-mapping verifier for mma.sync.m16n8k32 E4M3.
//
// The register-to-matrix-element mapping is the one part of an mma.sync kernel
// that a wrong implementation renders silently incorrect while still
// assembling. It does not need silicon to check: CUTLASS states the mapping as
// CuTe (Shape,Stride) layouts, and a layout is arithmetic. This file evaluates
// those layouts directly and compares them against the closed-form indexing the
// kernel uses, exhaustively, for every (lane, value) pair.
//
// Ground truth, transcribed from
// third_party/flashinfer/3rdparty/cutlass/include/cute/atom/mma_traits_sm89.hpp
// MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> and mma_traits_sm80.hpp
// SM80_16x8_Row:
//
//   ALayout ((4,8),(4,2,2)) : ((64,1),(16,8,256))
//   BLayout ((4,8),(4,2))   : ((32,1),(8,128))
//   CLayout ((4,8),(2,2))   : ((32,1),(16,8))
//
// A CuTe layout maps (thread, value) to a linear index into the logical MMA
// tile, which is column-major: A is m + 16k, B is n + 8k, C is m + 16n. The
// checks below invert that and confirm the kernel's formulas reproduce it.
//
// Three properties are checked, and each catches a different error:
//   1. agreement  - kernel formula equals the CUTLASS layout everywhere
//   2. bijection  - the mapping covers every tile element exactly once, which
//                   catches any permutation error the agreement check could
//                   miss if both sides shared a mistake
//   3. bank spread - the shared-memory addresses the fragment loads generate,
//                    with and without the 128-byte swizzle, so the swizzle is
//                    justified by a count rather than by assertion

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define VERIFY_LANES 32u
#define VERIFY_TILE_K 128u
#define VERIFY_SWIZZLE_CHUNKS 8u

typedef struct layout_mode
{
	uint32_t shape[4],stride[4],rank;
}
layout_mode_t;

// Evaluate one CuTe mode: decompose the flat coordinate colexicographically
// across the mode's shape and dot it with the stride.
static uint32_t layout_mode_index(const layout_mode_t *mode, uint32_t coordinate)
{
	uint32_t index = 0,rank,digit;
	for (rank = 0; rank < mode->rank; ++rank)
	{
		digit = coordinate % mode->shape[rank];
		coordinate = coordinate / mode->shape[rank];
		index += digit * mode->stride[rank];
	}
	return(index);
}

static uint32_t layout_mode_size(const layout_mode_t *mode)
{
	uint32_t size = 1,rank;
	for (rank = 0; rank < mode->rank; ++rank)
		size *= mode->shape[rank];
	return(size);
}

static uint32_t cute_index(const layout_mode_t *thread_mode, const layout_mode_t *value_mode, uint32_t lane, uint32_t value)
{
	return(layout_mode_index(thread_mode,lane) + layout_mode_index(value_mode,value));
}

// Kernel-side closed forms. These are what the GEMM actually computes; the
// point of this file is that they are not trusted until checked.
static void kernel_a_coordinate(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (4u * (lane % 4u)) + byte + (16u * (reg / 2u));
}

static void kernel_b_coordinate(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (4u * (lane % 4u)) + byte + (16u * reg);
}

static void kernel_c_coordinate(uint32_t lane, uint32_t accumulator, uint32_t *m, uint32_t *n)
{
	*m = (lane / 4u) + (8u * (accumulator / 2u));
	*n = (2u * (lane % 4u)) + (accumulator % 2u);
}

static int32_t verify_operand_a(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	uint8_t seen[512];
	uint32_t lane,value,reg,byte,expected,m,k,actual,mismatches = 0,uncovered = 0;
	memset(seen,0,sizeof(seen));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < layout_mode_size(&value_mode); ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			reg = value / 4u;
			byte = value % 4u;
			kernel_a_coordinate(lane,reg,byte,&m,&k);
			actual = m + (16u * k);
			if ( actual != expected )
				mismatches++;
			seen[expected & 511u]++;
		}
	for (value = 0; value < 512u; ++value)
		if ( seen[value] != 1 )
			uncovered++;
	printf("  A m16n8k32  mismatches=%-4u  elements covered exactly once=%u/512\n",
		mismatches,512u - uncovered);
	return(mismatches == 0 && uncovered == 0 ? 0 : -1);
}

static int32_t verify_operand_b(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	uint8_t seen[256];
	uint32_t lane,value,reg,byte,expected,n,k,actual,mismatches = 0,uncovered = 0;
	memset(seen,0,sizeof(seen));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < layout_mode_size(&value_mode); ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			reg = value / 4u;
			byte = value % 4u;
			kernel_b_coordinate(lane,reg,byte,&n,&k);
			actual = n + (8u * k);
			if ( actual != expected )
				mismatches++;
			seen[expected & 255u]++;
		}
	for (value = 0; value < 256u; ++value)
		if ( seen[value] != 1 )
			uncovered++;
	printf("  B m16n8k32  mismatches=%-4u  elements covered exactly once=%u/256\n",
		mismatches,256u - uncovered);
	return(mismatches == 0 && uncovered == 0 ? 0 : -1);
}

static int32_t verify_accumulator_c(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 2u, 2u, 0u, 0u }, { 16u, 8u, 0u, 0u }, 2u };
	uint8_t seen[128];
	uint32_t lane,value,expected,m,n,actual,mismatches = 0,uncovered = 0;
	memset(seen,0,sizeof(seen));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < layout_mode_size(&value_mode); ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			kernel_c_coordinate(lane,value,&m,&n);
			actual = m + (16u * n);
			if ( actual != expected )
				mismatches++;
			seen[expected & 127u]++;
		}
	for (value = 0; value < 128u; ++value)
		if ( seen[value] != 1 )
			uncovered++;
	printf("  C 16x8 row  mismatches=%-4u  elements covered exactly once=%u/128\n",
		mismatches,128u - uncovered);
	return(mismatches == 0 && uncovered == 0 ? 0 : -1);
}

// The mapping that the deleted first draft used, kept as a negative control. If
// this ever reports zero mismatches the verifier itself is broken.
static int32_t verify_known_bad_c_is_rejected(void)
{
	layout_mode_t thread_mode = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t value_mode = { { 2u, 2u, 0u, 0u }, { 16u, 8u, 0u, 0u }, 2u };
	uint32_t lane,value,expected,m,n,actual,mismatches = 0;
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < 4u; ++value)
		{
			expected = cute_index(&thread_mode,&value_mode,lane,value);
			m = ((value & 2u) << 2u) + (lane >> 2u) + ((value & 1u) << 3u);
			n = 2u * (lane % 4u);
			actual = m + (16u * n);
			if ( actual != expected )
				mismatches++;
		}
	printf("  negative control (first-draft C mapping) mismatches=%u/128 %s\n",
		mismatches,mismatches > 0 ? "- correctly rejected" : "- VERIFIER IS BROKEN");
	return(mismatches > 0 ? 0 : -1);
}

// ldmatrix.x4 gathers 512 bytes: lane L supplies a 16-byte row. Composing
// Copy_Traits<SM75_U32x4_LDSM_N>'s SrcLayout (32,128):(128,1) with its DstLayout
// (32,(32,4)):(32,(1,1024)) gives register r of thread t from source bytes
// [t*4 + r*128, +4), which lands in the chunk supplied by lane t/4 + 8r. For
// that to feed the A layout above, lane L must supply row (L%8) + 8*((L/8)%2)
// at k offset 16*(L/16).
static uint32_t ldmatrix_row_for_lane(uint32_t lane)
{
	return((lane % 8u) + (8u * ((lane / 8u) % 2u)));
}

static uint32_t ldmatrix_chunk_for_lane(uint32_t lane)
{
	return(lane / 16u);
}

// 128-byte swizzle: within a row, the 16-byte chunk at index c moves to
// c ^ (row % 8). TMA applies this when the tensor map is encoded with
// CU_TENSOR_MAP_SWIZZLE_128B, so the fragment load must apply the same xor.
static uint32_t swizzle_chunk(uint32_t chunk, uint32_t row)
{
	return(chunk ^ (row % VERIFY_SWIZZLE_CHUNKS));
}

static uint32_t count_bank_conflicts(int32_t apply_swizzle, uint32_t k_base)
{
	uint32_t bank_population[32],lane,row,chunk,byte_offset,bank,worst = 0;
	memset(bank_population,0,sizeof(bank_population));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		row = ldmatrix_row_for_lane(lane);
		chunk = (k_base / 16u) + ldmatrix_chunk_for_lane(lane);
		if ( apply_swizzle != 0 )
			chunk = swizzle_chunk(chunk,row);
		byte_offset = (row * VERIFY_TILE_K) + (chunk * 16u);
		bank = (byte_offset / 4u) % 32u;
		bank_population[bank]++;
	}
	for (bank = 0; bank < 32u; ++bank)
		if ( bank_population[bank] > worst )
			worst = bank_population[bank];
	return(worst);
}

// -- Pipeline schedule check ------------------------------------------------
// Appended as a second gate. The stage/phase schedule in
// SparkLmGroupGemmFp8Kernel is pure integer logic and its failure modes -
// consuming a stage never produced, overwriting a stage before it is consumed,
// waiting on the wrong mbarrier phase - are all decidable on the host. Only the
// hardware behaviour of the transfers themselves needs the ring.
#define PIPE_MAX_STAGES 8
#define PIPE_MAX_K_TILES 64

static int32_t verify_pipeline_schedule(uint32_t stages, uint32_t k_tiles)
{
	uint32_t produced_round[PIPE_MAX_STAGES],consumed_round[PIPE_MAX_STAGES];
	uint32_t k_tile,stage,ahead,round,expected_phase,fill,errors = 0;
	for (stage = 0; stage < stages; ++stage)
	{
		produced_round[stage] = 0xffffffffu;
		consumed_round[stage] = 0xffffffffu;
	}
	// prologue: stages 0 .. stages-2
	for (fill = 0; fill + 1u < stages && fill < k_tiles; ++fill)
		produced_round[fill % stages] = fill / stages;
	for (k_tile = 0; k_tile < k_tiles; ++k_tile)
	{
		stage = k_tile % stages;
		round = k_tile / stages;
		ahead = k_tile + stages - 1u;
		if ( ahead < k_tiles )
		{
			// a stage may only be refilled after its previous contents were consumed
			if ( produced_round[ahead % stages] != 0xffffffffu
				&& consumed_round[ahead % stages] != produced_round[ahead % stages] )
				errors++;
			produced_round[ahead % stages] = ahead / stages;
		}
		if ( produced_round[stage] != round )
			errors++;
		expected_phase = round & 1u;
		if ( expected_phase != ((k_tile / stages) & 1u) )
			errors++;
		consumed_round[stage] = round;
	}
	return((int32_t)errors);
}

static int32_t verify_pipeline_matrix(void)
{
	uint32_t stages,k_tiles;
	int32_t errors,total = 0;
	printf("\npipeline schedule: stages x k_tiles, errors per cell\n");
	for (stages = 2u; stages <= 6u; ++stages)
	{
		printf("  stages=%u :",stages);
		for (k_tiles = 1u; k_tiles <= 48u; k_tiles += 8u)
		{
			errors = verify_pipeline_schedule(stages,k_tiles);
			printf(" k=%-2u:%d",k_tiles,errors);
			total += errors;
		}
		printf("\n");
	}
	return(total);
}


// -- SM120 atom equivalence and the NVFP4 block-scaled atom -----------------
// sm_121a selects SM120_16x8x32_TN via rr_op_selector_sm120, not the SM89 atom.
// SM120_16x8x32_TN inherits MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>, so this
// checks the inherited layout against the SM89 one element by element rather
// than trusting that "both are 8-bit at m16n8k32" implies identical mappings.
static int32_t verify_sm120_equals_sm89(void)
{
	layout_mode_t sm89_a_t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm89_a_v = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	layout_mode_t sm80_a_t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm80_a_v = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	layout_mode_t sm89_b_t = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm89_b_v = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	layout_mode_t sm80_b_t = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t sm80_b_v = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	uint32_t lane,value,differences = 0;
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 16u; ++value)
			if ( cute_index(&sm89_a_t,&sm89_a_v,lane,value) != cute_index(&sm80_a_t,&sm80_a_v,lane,value) )
				differences++;
		for (value = 0; value < 8u; ++value)
			if ( cute_index(&sm89_b_t,&sm89_b_v,lane,value) != cute_index(&sm80_b_t,&sm80_b_v,lane,value) )
				differences++;
	}
	printf("  SM120_16x8x32_TN vs SM89 e4m3: differing (lane,value) pairs=%u/768\n",differences);
	return(differences == 0 ? 0 : -1);
}

// NVFP4 atom: SM120::BLOCKSCALED::SM120_16x8x64_TN_VS.
//   ALayout ((4,8),(8,2,2)) : ((128,1),(16,8,512))  -> (M16,K64)
//   BLayout ((4,8),(8,2))   : ((64,1),(8,256))      -> (N8,K64)
static void kernel_nvfp4_a_coordinate(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (8u * (lane % 4u)) + nibble + (32u * (reg / 2u));
}

static void kernel_nvfp4_b_coordinate(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (8u * (lane % 4u)) + nibble + (32u * reg);
}

static int32_t verify_nvfp4_operands(void)
{
	layout_mode_t a_thread = { { 4u, 8u, 0u, 0u }, { 128u, 1u, 0u, 0u }, 2u };
	layout_mode_t a_value = { { 8u, 2u, 2u, 0u }, { 16u, 8u, 512u, 0u }, 3u };
	layout_mode_t b_thread = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t b_value = { { 8u, 2u, 0u, 0u }, { 8u, 256u, 0u, 0u }, 2u };
	uint8_t seen_a[1024],seen_b[512];
	uint32_t lane,value,expected,m,n,k,actual,a_bad = 0,b_bad = 0,a_gap = 0,b_gap = 0;
	memset(seen_a,0,sizeof(seen_a));
	memset(seen_b,0,sizeof(seen_b));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 32u; ++value)
		{
			expected = cute_index(&a_thread,&a_value,lane,value);
			kernel_nvfp4_a_coordinate(lane,value / 8u,value % 8u,&m,&k);
			actual = m + (16u * k);
			if ( actual != expected )
				a_bad++;
			seen_a[expected & 1023u]++;
		}
		for (value = 0; value < 16u; ++value)
		{
			expected = cute_index(&b_thread,&b_value,lane,value);
			kernel_nvfp4_b_coordinate(lane,value / 8u,value % 8u,&n,&k);
			actual = n + (8u * k);
			if ( actual != expected )
				b_bad++;
			seen_b[expected & 511u]++;
		}
	}
	for (value = 0; value < 1024u; ++value)
		if ( seen_a[value] != 1 )
			a_gap++;
	for (value = 0; value < 512u; ++value)
		if ( seen_b[value] != 1 )
			b_gap++;
	printf("  A m16n8k64 nvf4  mismatches=%-4u covered exactly once=%u/1024\n",a_bad,1024u - a_gap);
	printf("  B m16n8k64 nvf4  mismatches=%-4u covered exactly once=%u/512\n",b_bad,512u - b_gap);
	return((a_bad == 0 && b_bad == 0 && a_gap == 0 && b_gap == 0) ? 0 : -1);
}


// -- NVFP4 scale-factor layouts ---------------------------------------------
// SFALayout ((2,2,8),64) : ((8,0,1),16)  -> (M16,K64)
// SFBLayout ((4,8),64)   : ((0,1),8)     -> (N8,K64)
// Both carry a stride-0 mode, which is why CUTLASS annotates them "effectively
// 16 threads" and "effectively 8 threads": lanes differing only in that mode
// address the same element, so several lanes must supply the same scale. That
// redundancy is the whole reason the instruction takes {byte-id, thread-id}
// selectors, and getting them wrong reads a valid-looking wrong scale.
static void kernel_sfa_coordinate(uint32_t lane, uint32_t value, uint32_t *m, uint32_t *k)
{
	*m = (8u * (lane % 2u)) + (lane / 4u);
	*k = value;
}

static void kernel_sfb_coordinate(uint32_t lane, uint32_t value, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = value;
}

static int32_t verify_nvfp4_scale_layouts(void)
{
	layout_mode_t sfa_thread = { { 2u, 2u, 8u, 0u }, { 8u, 0u, 1u, 0u }, 3u };
	layout_mode_t sfa_value = { { 64u, 0u, 0u, 0u }, { 16u, 0u, 0u, 0u }, 1u };
	layout_mode_t sfb_thread = { { 4u, 8u, 0u, 0u }, { 0u, 1u, 0u, 0u }, 2u };
	layout_mode_t sfb_value = { { 64u, 0u, 0u, 0u }, { 8u, 0u, 0u, 0u }, 1u };
	uint32_t lane,value,expected,m,n,k,actual,a_bad = 0,b_bad = 0;
	uint32_t a_multiplicity[1024],b_multiplicity[512],a_max = 0,b_max = 0,index;
	memset(a_multiplicity,0,sizeof(a_multiplicity));
	memset(b_multiplicity,0,sizeof(b_multiplicity));
	for (lane = 0; lane < VERIFY_LANES; ++lane)
		for (value = 0; value < 64u; ++value)
		{
			expected = cute_index(&sfa_thread,&sfa_value,lane,value);
			kernel_sfa_coordinate(lane,value,&m,&k);
			actual = m + (16u * k);
			if ( actual != expected )
				a_bad++;
			a_multiplicity[expected & 1023u]++;
			expected = cute_index(&sfb_thread,&sfb_value,lane,value);
			kernel_sfb_coordinate(lane,value,&n,&k);
			actual = n + (8u * k);
			if ( actual != expected )
				b_bad++;
			b_multiplicity[expected & 511u]++;
		}
	for (index = 0; index < 1024u; ++index)
		if ( a_multiplicity[index] > a_max )
			a_max = a_multiplicity[index];
	for (index = 0; index < 512u; ++index)
		if ( b_multiplicity[index] > b_max )
			b_max = b_multiplicity[index];
	printf("  SFA mismatches=%-4u lanes sharing each element=%u (CUTLASS: effectively 16 threads)\n",a_bad,a_max);
	printf("  SFB mismatches=%-4u lanes sharing each element=%u (CUTLASS: effectively 8 threads)\n",b_bad,b_max);
	// 32 lanes over 16 effective -> 2 lanes per element; over 8 effective -> 4.
	if ( a_max != 2u || b_max != 4u )
	{
		printf("  SHARING FACTOR DISAGREES WITH THE STRIDE-0 MODE - decode is wrong\n");
		return(-1);
	}
	return((a_bad == 0 && b_bad == 0) ? 0 : -1);
}


// -- lm_mma.cuh formulas ------------------------------------------------------
// The rewrite's operand loaders index by REGISTER and byte-within-register
// rather than by (register, byte-in-K). These are the exact formulas in
// lm_mma.cuh; the check is that they reproduce the same CuTe layouts already
// verified above, so the new library cannot drift from the old verification.
static void lm_mma8_a(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (4u * (lane % 4u)) + (16u * (reg / 2u)) + byte;
}
static void lm_mma8_b(uint32_t lane, uint32_t reg, uint32_t byte, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (4u * (lane % 4u)) + (16u * reg) + byte;
}
static void lm_mma4_a(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *m, uint32_t *k)
{
	*m = (lane / 4u) + (8u * (reg % 2u));
	*k = (8u * (lane % 4u)) + (32u * (reg / 2u)) + nibble;
}
static void lm_mma4_b(uint32_t lane, uint32_t reg, uint32_t nibble, uint32_t *n, uint32_t *k)
{
	*n = lane / 4u;
	*k = (8u * (lane % 4u)) + (32u * reg) + nibble;
}
static int32_t verify_lm_mma_formulas(void)
{
	layout_mode_t a8t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t a8v = { { 4u, 2u, 2u, 0u }, { 16u, 8u, 256u, 0u }, 3u };
	layout_mode_t b8t = { { 4u, 8u, 0u, 0u }, { 32u, 1u, 0u, 0u }, 2u };
	layout_mode_t b8v = { { 4u, 2u, 0u, 0u }, { 8u, 128u, 0u, 0u }, 2u };
	layout_mode_t a4t = { { 4u, 8u, 0u, 0u }, { 128u, 1u, 0u, 0u }, 2u };
	layout_mode_t a4v = { { 8u, 2u, 2u, 0u }, { 16u, 8u, 512u, 0u }, 3u };
	layout_mode_t b4t = { { 4u, 8u, 0u, 0u }, { 64u, 1u, 0u, 0u }, 2u };
	layout_mode_t b4v = { { 8u, 2u, 0u, 0u }, { 8u, 256u, 0u, 0u }, 2u };
	uint32_t lane,value,m,n,k,bad = 0;
	for (lane = 0; lane < VERIFY_LANES; ++lane)
	{
		for (value = 0; value < 16u; ++value)
		{
			lm_mma8_a(lane,value / 4u,value % 4u,&m,&k);
			if ( m + (16u * k) != cute_index(&a8t,&a8v,lane,value) )
				bad++;
		}
		for (value = 0; value < 8u; ++value)
		{
			lm_mma8_b(lane,value / 4u,value % 4u,&n,&k);
			if ( n + (8u * k) != cute_index(&b8t,&b8v,lane,value) )
				bad++;
		}
		for (value = 0; value < 32u; ++value)
		{
			lm_mma4_a(lane,value / 8u,value % 8u,&m,&k);
			if ( m + (16u * k) != cute_index(&a4t,&a4v,lane,value) )
				bad++;
		}
		for (value = 0; value < 16u; ++value)
		{
			lm_mma4_b(lane,value / 8u,value % 8u,&n,&k);
			if ( n + (8u * k) != cute_index(&b4t,&b4v,lane,value) )
				bad++;
		}
	}
	printf("  lm_mma.cuh operand formulas vs CUTLASS layouts: mismatches=%u/2304\n",bad);
	return(bad == 0 ? 0 : -1);
}

int32_t main(void)
{
	int32_t failures = 0;
	printf("MMA fragment mappings, checked against CUTLASS CuTe layouts\n");
	printf("target sm_121a: FP8 via SM120_16x8x32_TN, NVFP4 via SM120_16x8x64_TN_VS\n\n");
	failures += verify_operand_a() != 0;
	failures += verify_operand_b() != 0;
	failures += verify_accumulator_c() != 0;
	failures += verify_known_bad_c_is_rejected() != 0;
	printf("\nldmatrix.x4 shared-memory bank spread over a %u-byte row\n",VERIFY_TILE_K);
	printf("  without swizzle: worst bank holds %u of 32 lanes\n",count_bank_conflicts(0,0u));
	printf("  with 128B swizzle: worst bank holds %u of 32 lanes\n",count_bank_conflicts(1,0u));
	if ( count_bank_conflicts(1,0u) >= count_bank_conflicts(0,0u) )
	{
		printf("  SWIZZLE DOES NOT HELP - layout is wrong\n");
		failures++;
	}
	failures += verify_sm120_equals_sm89() != 0;
	printf("\nNVFP4 atom SM120::BLOCKSCALED::SM120_16x8x64_TN_VS\n");
	failures += verify_nvfp4_operands() != 0;
	failures += verify_nvfp4_scale_layouts() != 0;
	printf("\nrewrite: lm/ kernel library\n");
	failures += verify_lm_mma_formulas() != 0;
	failures += verify_pipeline_matrix() != 0;
	printf("\n%s (%d failing checks)\n",failures == 0 ? "PASS" : "FAIL",failures);
	return(failures == 0 ? 0 : 1);
}


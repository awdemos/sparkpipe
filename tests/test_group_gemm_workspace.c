// NVFP4 routed-MoE workspace layout, checked without CUDA.
// The failure guarded against: two regions overlapping, which corrupts data
// with no allocation error anywhere.
#include "sparkpipe/spark_lm_group_gemm_workspace.h"
#include <stdio.h>
#include <string.h>
static int32_t fails=0;
static void ck(int c,const char*l){printf(c?"  ok   %s\n":"  FAIL %s\n",l); if(!c)fails++;}
int main(void){
	spark_lm_workspace_shape_t shape; spark_lm_workspace_layout_t layout;
	uint32_t a,b; int32_t st;
	memset(&shape,0,sizeof shape);
	shape.tokens=128; shape.top_k=8; shape.expert_count=256;
	shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
	shape.tile_m=16; shape.tile_n=128;
	printf("GLM 5.2 routed MoE, B128, NVFP4 workspace\n\n");
	st=spark_lm_workspace_layout_build(&shape,&layout);
	ck(st==SPARK_LM_WORKSPACE_OK,"builds");
	ck(layout.packed_rows==1024,"128 tokens x top-8 = 1024 packed rows");
	ck(layout.bytes[SPARK_LM_WORKSPACE_REGION_PACKED_HIDDEN]==1024ULL*3072ULL,
	   "packed hidden is rows x hidden/2 bytes (NVFP4), not rows x hidden");
	ck(layout.bytes[SPARK_LM_WORKSPACE_REGION_PACKED_HIDDEN_SCALE]==1024ULL*384ULL,
	   "hidden scales are one UE4M3 byte per 16 elements");
	ck(layout.bytes[SPARK_LM_WORKSPACE_REGION_GATE_UP_BF16]==1024ULL*2048ULL*2ULL*2ULL,
	   "gate+up bf16 carries both components");
	ck(layout.bytes[SPARK_LM_WORKSPACE_REGION_INTERMEDIATE]==1024ULL*1024ULL,
	   "intermediate is rows x intermediate/2 bytes");
	printf("\nno region overlaps its neighbour, all aligned\n");
	for(a=0;a<SPARK_LM_WORKSPACE_REGION_COUNT;a++){
		if((layout.offset[a]%SPARK_LM_WORKSPACE_ALIGNMENT)!=0){
			printf("  FAIL region %u misaligned\n",a); fails++; }
		for(b=a+1;b<SPARK_LM_WORKSPACE_REGION_COUNT;b++){
			uint64_t ae=layout.offset[a]+layout.bytes[a], be=layout.offset[b]+layout.bytes[b];
			if(layout.offset[a]<be && layout.offset[b]<ae){
				printf("  FAIL regions %u and %u overlap\n",a,b); fails++; }}}
	ck(1,"9 regions pairwise disjoint and 256-byte aligned");
	ck(layout.total_bytes>=layout.offset[SPARK_LM_WORKSPACE_REGION_ROUTE_OUTPUT_BF16]
	   +layout.bytes[SPARK_LM_WORKSPACE_REGION_ROUTE_OUTPUT_BF16],
	   "total covers the last region");
	printf("\ntile count reflects that decode groups are short\n");
	printf("  packed rows %llu over %u experts -> %llu tiles\n",
	   (unsigned long long)layout.packed_rows,shape.expert_count,
	   (unsigned long long)layout.total_tiles);
	ck(layout.total_tiles==256ULL*1ULL*32ULL,
	   "4 rows/expert fits one M tile; 4096 N over TILE_N=128 is 32");
	printf("\nTILE_M selection across token buckets\n");
	printf("  the failure this prevents: a tile shorter than an expert's row count\n");
	printf("  splits every expert in two, and each M tile re-reads the same weight\n");
	printf("  tile, doubling the stream that is 96%% of all traffic.\n");
	{
		static const uint32_t buckets[] = { 1u, 8u, 64u, 128u, 512u, 1024u, 2048u };
		uint32_t i;
		for (i = 0; i < 7u; ++i)
		{
			uint64_t peak; uint64_t m_tiles;
			memset(&shape,0,sizeof shape);
			shape.tokens=buckets[i]; shape.top_k=8; shape.expert_count=256;
			shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
			shape.tile_m=0; shape.tile_n=128;
			if (spark_lm_workspace_layout_build(&shape,&layout)!=SPARK_LM_WORKSPACE_OK){
				printf("  FAIL bucket %u did not build\n",buckets[i]); fails++; continue; }
			peak = spark_lm_workspace_peak_rows_per_expert(&shape);
			m_tiles = (peak + layout.tile_m - 1u) / layout.tile_m;
			printf("    B%-5u peak rows/expert %-4llu -> TILE_M %-4u  M tiles %llu %s\n",
				buckets[i],(unsigned long long)peak,layout.tile_m,
				(unsigned long long)m_tiles, m_tiles<=1?"":"  <-- WEIGHT RE-READ");
			if (m_tiles > 1u){ fails++; }
		}
	}
	ck(1,"every bucket resolves to a single M tile, so no weight tile is read twice");
	printf("\n  stage depth selected with the tile, bounded by 128 KB shared\n");
	{
		static const uint32_t buckets[] = { 1u, 128u, 512u, 1024u, 2048u };
		uint32_t i;
		for (i = 0; i < 5u; ++i)
		{
			memset(&shape,0,sizeof shape);
			shape.tokens=buckets[i]; shape.top_k=8; shape.expert_count=256;
			shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
			shape.tile_m=0; shape.tile_n=128;
			if (spark_lm_workspace_layout_build(&shape,&layout)!=SPARK_LM_WORKSPACE_OK){
				printf("    FAIL B%u did not build\n",buckets[i]); fails++; continue; }
			printf("    B%-5u TILE_M %-4u stages %-2u shared %6llu B %s\n",
				buckets[i],layout.tile_m,layout.stages,
				(unsigned long long)layout.shared_bytes,
				layout.shared_bytes<=131072ULL?"":"  <-- OVER LIMIT");
			if (layout.shared_bytes > 131072ULL) fails++;
		}
	}
	ck(1,"every selected (TILE_M, stages) pair fits in shared memory");
	{
		/* the regression: TILE_M=128 with the four stages that were hardcoded
		   needs 131136 bytes and the launch fails outright. */
		ck(spark_lm_workspace_shared_bytes(128u,128u,256u,4u,4u) > 131072ULL,
		   "TILE_M=128 at 4 stages provably exceeds shared memory");
		ck(spark_lm_workspace_shared_bytes(128u,128u,256u,3u,4u) <= 131072ULL,
		   "TILE_M=128 at 3 stages fits, which is why stages track the tile");
		ck(spark_lm_workspace_select_stages(16u,128u,256u,4u,131072ULL)==6u,
		   "a 16-row tile affords six stages, the depth the weight stream wants");
	}
	{
		/* the regression this guards: TILE_M pinned at 16 doubles B1024 */
		memset(&shape,0,sizeof shape);
		shape.tokens=1024; shape.top_k=8; shape.expert_count=256;
		shape.hidden_dimension=6144; shape.intermediate_dimension=2048;
		shape.tile_m=16; shape.tile_n=128;
		spark_lm_workspace_layout_build(&shape,&layout);
		ck(layout.tile_m==16,"an explicit TILE_M is honoured, so a sweep can pin it");
		shape.tile_m=0;
		spark_lm_workspace_layout_build(&shape,&layout);
		ck(layout.tile_m==64,"B1024 auto-selects 64, not the 16 that would double the stream");
	}
	printf("\nrejections\n");
	memset(&shape,0,sizeof shape);
	shape.tokens=128; shape.top_k=8; shape.expert_count=256;
	shape.intermediate_dimension=2048; shape.tile_m=16; shape.tile_n=128;
	shape.hidden_dimension=6150;
	ck(spark_lm_workspace_layout_build(&shape,&layout)==SPARK_LM_WORKSPACE_ERR_GROUP,
	   "hidden not divisible by the NVFP4 group is rejected");
	shape.hidden_dimension=6144; shape.intermediate_dimension=0;
	ck(spark_lm_workspace_layout_build(&shape,&layout)==SPARK_LM_WORKSPACE_ERR_SHAPE,
	   "zero intermediate rejected");
	printf("\n%s (%d failing)\n",fails?"FAIL":"PASS",fails); return fails?1:0;}

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
	printf("\nrejections\n");
	shape.hidden_dimension=6150;
	ck(spark_lm_workspace_layout_build(&shape,&layout)==SPARK_LM_WORKSPACE_ERR_GROUP,
	   "hidden not divisible by the NVFP4 group is rejected");
	shape.hidden_dimension=6144; shape.intermediate_dimension=0;
	ck(spark_lm_workspace_layout_build(&shape,&layout)==SPARK_LM_WORKSPACE_ERR_SHAPE,
	   "zero intermediate rejected");
	printf("\n%s (%d failing)\n",fails?"FAIL":"PASS",fails); return fails?1:0;}

#include "sparkpipe/spark_lm_tensor_map_encode.h"
#include <stdio.h>
uint32_t stub_rank(void); uint64_t stub_dim(int); uint32_t stub_box(int);
uint64_t stub_stride(int); int stub_swizzle(void);
static int fails=0;
static void ck(int c,const char*l){printf(c?"  ok   %s\n":"  FAIL %s\n",l); if(!c)fails++;}
int main(void){
  static uint8_t buf[512] __attribute__((aligned(128)));
  CUtensorMap map; spark_lm_tensor_map_request_t rq; int32_t st;
  printf("cuTensorMapEncodeTiled argument marshalling\n\nNVFP4 weights 256x4096x6144, TILE 128x256\n");
  memset(&rq,0,sizeof rq);
  rq.global_address=buf; rq.rows=4096; rq.columns=6144; rq.groups=256;
  rq.box_rows=128; rq.box_columns=256; rq.element_bits=SPARK_LM_TENSOR_MAP_BITS_NVFP4;
  st=spark_lm_tensor_map_prepare(&map,&rq);
  ck(st==SPARK_LM_TENSOR_MAP_ENCODE_OK,"prepare succeeds");
  ck(stub_rank()==3,"rank 3 reaches the driver");
  ck(stub_dim(0)==3072,"innermost dim is K/2 BYTES not K elements");
  ck(stub_box(0)==128,"box inner extent is 128 bytes");
  ck(stub_stride(0)==3072,"globalStrides[0] is the row pitch (dim1 stride)");
  ck(stub_stride(1)==3072ULL*4096ULL,"globalStrides[1] is the expert stride");
  ck(stub_swizzle()==CU_TENSOR_MAP_SWIZZLE_128B,"128B swizzle selected");
  printf("\nFP8 activations 128x6144, TILE 16x128\n");
  memset(&rq,0,sizeof rq);
  rq.global_address=buf; rq.rows=128; rq.columns=6144; rq.groups=1;
  rq.box_rows=16; rq.box_columns=128; rq.element_bits=SPARK_LM_TENSOR_MAP_BITS_FP8;
  st=spark_lm_tensor_map_prepare(&map,&rq);
  ck(st==SPARK_LM_TENSOR_MAP_ENCODE_OK,"prepare succeeds");
  ck(stub_rank()==2,"rank 2 for a single group");
  ck(stub_dim(0)==6144,"FP8 innermost dim equals K");
  printf("\ngeometry rejections still fire before the driver is touched\n");
  memset(&rq,0,sizeof rq);
  rq.global_address=buf; rq.rows=4096; rq.columns=6144; rq.groups=256;
  rq.box_rows=128; rq.box_columns=128; rq.element_bits=SPARK_LM_TENSOR_MAP_BITS_NVFP4;
  ck(spark_lm_tensor_map_prepare(&map,&rq)==SPARK_LM_TENSOR_MAP_ERR_BOX_ALIGN,
     "NVFP4 TILE_K=128 rejected before encode");
  printf("\n%s (%d failing)\n",fails?"FAIL":"PASS",fails); return fails?1:0;}

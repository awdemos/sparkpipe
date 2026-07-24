#include "sparkpipe/spark_glm52_tp_shard.h"

#include <assert.h>
#include <string.h>

// Real GLM-5.2 shapes: hidden 6144, 64 heads, q head block 192+64=256, kv_b
// head block 192+256=448, o_proj input block 256, dense intermediate 12288,
// shared-expert intermediate 2048, latent+rope 576.

static void SparkTestTpShardSpec(SparkGlm52StagePackTensorSpec *spec,const char *name,uint64_t dim0,uint64_t dim1)
{
	memset(spec,0,sizeof(*spec));
	spec->abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	spec->rank = 2u;
	spec->bytes_per_element = 2u;
	spec->shape[0] = dim0;
	spec->shape[1] = dim1;
	spec->tensor_name = name;
	spec->dtype = "BF16";
}

static void SparkTestTpShardShape(SparkGlm52TpShapeDescriptor *shape,uint32_t degree,uint32_t rank)
{
	memset(shape,0,sizeof(*shape));
	shape->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	shape->tp_degree = degree;
	shape->tp_rank = rank;
	shape->pp_stage_count = 1u;
	shape->pp_stage_index = 0u;
}

static void SparkTestTpShardGeometry(SparkGlm52TpModelGeometry *geometry)
{
	memset(geometry,0,sizeof(*geometry));
	geometry->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	geometry->head_count = 64u;
	geometry->q_b_head_block = 256u;
	geometry->kv_b_head_block = 448u;
	geometry->o_proj_head_block = 256u;
}

static void SparkTestTpShardClassification(void)
{
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.q_b_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.kv_b_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.o_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM_HEADS);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.0.mlp.gate_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.9.mlp.shared_experts.up_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.0.mlp.down_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.9.mlp.shared_experts.down_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.q_a_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.kv_a_proj_with_mqa.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.mlp.gate.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.embed_tokens.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.mystery.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_UNKNOWN);
}

// The four tp4 shards of q_b tile the output dimension exactly: contiguous,
// non-overlapping, head-block aligned, and their bytes sum to the full tensor.
static void SparkTestTpShardTilingExactness(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpModelGeometry geometry;
	uint64_t next_offset,total_bytes,full_bytes;
	uint32_t rank_index;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.q_b_proj.weight",16384u,1536u);
	SparkTestTpShardGeometry(&geometry);
	full_bytes = 16384u * 1536u * 2u;
	next_offset = 0u;
	total_bytes = 0u;
	for (rank_index = 0u; rank_index < 4u; ++rank_index)
	{
		SparkGlm52TpShapeDescriptor shape;
		SparkGlm52TpShardView view;
		SparkTestTpShardShape(&shape,4u,rank_index);
		assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
		assert(view.split_dimension == 0u);
		assert(view.element_offset == next_offset);
		assert(view.element_extent == 4096u);
		assert(view.element_offset % 256u == 0u);
		next_offset = view.element_offset + view.element_extent;
		total_bytes += view.shard_bytes;
	}
	assert(next_offset == 16384u);
	assert(total_bytes == full_bytes);
}

// o_proj splits its input dimension on value-head blocks; tp8 gives eight
// 2048-element input slices.
static void SparkTestTpShardInputDimHeads(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.o_proj.weight",6144u,16384u);
	SparkTestTpShardShape(&shape,8u,5u);
	SparkTestTpShardGeometry(&geometry);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
	assert(view.split_dimension == 1u);
	assert(view.element_extent == 2048u);
	assert(view.element_offset == 5u * 2048u);
	assert(view.shard_bytes == 6144u * 2048u * 2u);
}

// Replicated tensors load whole on every rank; the latent kv_a path is the
// canonical case.
static void SparkTestTpShardReplicated(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.kv_a_proj_with_mqa.weight",576u,6144u);
	SparkTestTpShardShape(&shape,4u,2u);
	SparkTestTpShardGeometry(&geometry);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
	assert(view.shard_class == SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(view.element_offset == 0u);
	assert(view.shard_bytes == 576u * 6144u * 2u);
}

// Degree one is a whole-tensor view for every class including unknown, so
// existing single-shape packs keep loading unchanged.
static void SparkTestTpShardDegreeOneCompat(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"some.future.tensor.weight",100u,200u);
	SparkTestTpShardShape(&shape,1u,0u);
	SparkTestTpShardGeometry(&geometry);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
	assert(view.shard_bytes == 100u * 200u * 2u);
}

static void SparkTestTpShardFailsClosed(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"model.layers.0.mlp.gate_proj.weight",12288u,6144u);
	SparkTestTpShardGeometry(&geometry);
	// Degrees that do not divide the model are rejected outright.
	SparkTestTpShardShape(&shape,3u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	SparkTestTpShardShape(&shape,13u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Rank out of range.
	SparkTestTpShardShape(&shape,4u,4u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Unknown tensors at any real degree fail closed instead of guessing.
	SparkTestTpShardSpec(&spec,"model.layers.0.mystery.weight",12288u,6144u);
	SparkTestTpShardShape(&shape,2u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_VALIDATION_FAILED);
	// A dimension the degree does not divide is rejected.
	SparkTestTpShardSpec(&spec,"model.layers.0.mlp.gate_proj.weight",12290u,6144u);
	SparkTestTpShardShape(&shape,8u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Head count that the degree does not divide is rejected at validation.
	SparkTestTpShardSpec(&spec,"model.layers.0.mlp.gate_proj.weight",12288u,6144u);
	SparkTestTpShardShape(&shape,4u,0u);
	geometry.head_count = 6u;
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
}

// The contract hash separates every degree, rank, and tensor, and is stable
// for identical inputs.
static void SparkTestTpShardGeometryHash(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape_a,shape_b;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view_a,view_b;
	uint64_t hash_a,hash_b,hash_a_repeat;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.q_b_proj.weight",16384u,1536u);
	SparkTestTpShardGeometry(&geometry);
	SparkTestTpShardShape(&shape_a,4u,1u);
	SparkTestTpShardShape(&shape_b,4u,2u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape_a,&geometry,&view_a) == SPARK_STATUS_OK);
	assert(SparkGlm52TpShardComputeView(&spec,&shape_b,&geometry,&view_b) == SPARK_STATUS_OK);
	hash_a = SparkGlm52TpShardGeometryHash(&spec,&shape_a,&view_a);
	hash_b = SparkGlm52TpShardGeometryHash(&spec,&shape_b,&view_b);
	hash_a_repeat = SparkGlm52TpShardGeometryHash(&spec,&shape_a,&view_a);
	assert(hash_a != 0u && hash_b != 0u);
	assert(hash_a != hash_b);
	assert(hash_a == hash_a_repeat);
	SparkTestTpShardShape(&shape_b,2u,1u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape_b,&geometry,&view_b) == SPARK_STATUS_OK);
	assert(SparkGlm52TpShardGeometryHash(&spec,&shape_b,&view_b) != hash_a);
}

int main(void)
{
	SparkTestTpShardClassification();
	SparkTestTpShardTilingExactness();
	SparkTestTpShardInputDimHeads();
	SparkTestTpShardReplicated();
	SparkTestTpShardDegreeOneCompat();
	SparkTestTpShardFailsClosed();
	SparkTestTpShardGeometryHash();
	return 0;
}

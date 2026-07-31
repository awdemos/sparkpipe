// qwen36's stage doorway: the three model-validation externs the common
// stage module resolves at link. Fail-closed until the family's execute
// path lands - generic contract checks pass, everything model-specific
// refuses, and nothing half-runs silently.
#include <stdint.h>
#include "sparkpipe/spark_resident_decode_stage.h"

SparkStatus SparkResidentDecodeStageModelValidateNodeContext(const SparkResidentDecodeStageNodeContext *node_context)
{
	if ( node_context == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( node_context->abi_version != SPARK_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkResidentDecodeStageModelValidateSliceNodeContext(const SparkResidentDecodeStageSliceNodeContext *slice_node_context, const SparkResidentDecodeStageNodeContext **first_node_context)
{
	if ( slice_node_context == 0 || first_node_context == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkResidentDecodeStageModelValidateFrameTaps(const SparkResidentDecodeStageFrameContext *frame_context)
{
	if ( frame_context == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( (frame_context->flags & SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MODEL_HIDDEN_TAPS) != 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

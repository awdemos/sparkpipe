// The bring-up admission proof: a family with no measured cost profile
// - qwen 3.6's geometry here, dense hybrid, sixty-four layers, no
// routed boundary - initializes the scheduler on the uniform estimated
// profile and gets a decode request ADMITTED with a thirteen-stage
// balanced plan. This is the K2 gap closed generically: K3 and any
// future family ride the same path until their measurements exist.
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_scheduler.h"
#include "sparkpipe/spark_stage_plan.h"

int main(void)
{
	SparkScheduler scheduler;
	SparkSchedulerConfiguration configuration;
	SparkSchedulerRequest request;
	SparkSchedulerDecision decision;
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_SCHEDULER_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.spark_count = SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT;
	configuration.queue_depth_per_spark = 2u;
	configuration.measured_profile_id =
		SPARK_STAGE_PLAN_PROFILE_UNIFORM_ESTIMATED;
	configuration.quantization_mode = SPARK_STAGE_PLAN_QUANTIZATION_AUTO;
	configuration.configuration_flags =
		SPARK_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL;
	configuration.prefix_cache_block_tokens = 64u;
	// Dense hybrid: no routed layers, so the routed boundary sits at
	// the end and routed accounting is zero everywhere.
	configuration.stage_geometry.layer_count = 64u;
	configuration.stage_geometry.first_routed_layer = 64u;
	// ~4.2 GB of bf16 weights per stage at 273 GB/s: ~15.4 ms, so
	// ~240 us per layer is the honest scale of the estimate.
	configuration.estimated_layer_cost_ns = 240000u;
	configuration.estimated_final_stage_extra_cost_ns = 2000000u;
	assert(SparkSchedulerInitialize(&scheduler, &configuration) ==
		SPARK_STATUS_OK);
	memset(&request, 0, sizeof(request));
	request.abi_version = SPARK_SCHEDULER_ABI_VERSION;
	request.descriptor_bytes = (uint32_t)sizeof(request);
	request.flags = SPARK_SCHEDULER_REQUEST_FLAG_DECODE;
	request.sequence_id = 1u;
	request.active_sequence_count = 8u;
	request.prompt_token_count = 1024u;
	request.computed_prompt_token_count = 1024u;
	memset(&decision, 0, sizeof(decision));
	assert(SparkSchedulerAdmit(&scheduler, &request, &decision) ==
		SPARK_STATUS_OK);
	assert(decision.accepted == 1u);
	assert(decision.stage_count == SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT);
	assert(decision.stage_plan.stages[0u].layer_count != 0u);
	return 0;
}

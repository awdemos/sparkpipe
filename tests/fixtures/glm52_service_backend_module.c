#include <string.h>

#include "sparkpipe/spark_glm52_service_backend.h"

static SparkStatus SparkTestServiceBackendInitialize(
	const SparkGlm52ServiceBackendConfiguration *configuration,
	void **backend_state)
{
	static uint32_t State;

	if (configuration == 0 || backend_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->abi_version != SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	*backend_state = &State;
	return SPARK_STATUS_OK;
}

static void SparkTestServiceBackendDestroy(void *backend_state)
{
	(void)backend_state;
}

static SparkStatus SparkTestServiceBackendGetView(
	void *backend_state,
	SparkGlm52ServiceBackendView *view)
{
	if (backend_state == 0 || view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES;
	view->backend_ready = 1u;
	view->pp13_ready = 1u;
	view->max_context_tokens = 1048576u;
	view->production_contract_flags =
		SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
	view->service = (SparkGlm52ServiceRuntime *)(uintptr_t)0x1000u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServiceBackendPump(
	void *backend_state,
	uint32_t max_dispatch_steps,
	SparkGlm52ServiceStats *stats_out)
{
	if (backend_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (stats_out != 0)
		memset(stats_out,0,sizeof(*stats_out));
	(void)max_dispatch_steps;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServiceBackendGetPollDescriptors(
	void *backend_state,
	SparkGlm52ServiceBackendPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out)
{
	if (backend_state == 0 || descriptor_count_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	(void)descriptors;
	(void)descriptor_capacity;
	*descriptor_count_out = 0u;
	return SPARK_STATUS_OK;
}

static const SparkGlm52ServiceBackendInterface SparkTestServiceBackendInterface =
{
	SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION,
	SPARK_GLM52_SERVICE_BACKEND_INTERFACE_BYTES,
	SPARK_GLM52_SERVICE_BACKEND_REQUIRED_PRODUCTION_CAPS,
	0u,
	SparkTestServiceBackendInitialize,
	SparkTestServiceBackendDestroy,
	SparkTestServiceBackendGetView,
	SparkTestServiceBackendPump,
	SparkTestServiceBackendGetPollDescriptors
};

const SparkGlm52ServiceBackendInterface *SparkGlm52ServiceBackendGetInterface(void)
{
	return &SparkTestServiceBackendInterface;
}

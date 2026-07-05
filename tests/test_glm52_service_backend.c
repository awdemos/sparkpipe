#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_service_backend.h"

#ifndef TEST_SERVICE_BACKEND_MODULE_PATH
#define TEST_SERVICE_BACKEND_MODULE_PATH "build/test_modules/libglm52_service_backend_module.so"
#endif

static void SparkTestServiceBackendLoadsRequiredInterface(void)
{
	SparkGlm52ServiceBackendDynamicLibrary library;
	SparkGlm52ServiceBackendConfiguration configuration;
	SparkGlm52ServiceBackendView view;
	void *backend_state;

	memset(&library,0,sizeof(library));
	assert(SparkGlm52ServiceBackendLoadInterfaceFromSharedObject(
		TEST_SERVICE_BACKEND_MODULE_PATH,
		SPARK_GLM52_SERVICE_BACKEND_REQUIRED_PRODUCTION_CAPS,
		&library) == SPARK_STATUS_OK);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES;
	backend_state = 0;
	assert(library.backend_interface.initialize(
		&configuration,
		&backend_state) == SPARK_STATUS_OK);
	assert(backend_state != 0);
	memset(&view,0,sizeof(view));
	assert(library.backend_interface.get_view(backend_state,&view) ==
		SPARK_STATUS_OK);
	assert(view.service != 0);
	assert(view.backend_ready == 1u);
	assert(view.pp13_ready == 1u);
	library.backend_interface.destroy(backend_state);
	SparkGlm52ServiceBackendUnloadInterface(&library);
}

int main(void)
{
	SparkTestServiceBackendLoadsRequiredInterface();
	return 0;
}

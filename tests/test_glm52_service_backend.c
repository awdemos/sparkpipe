#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_service_backend.h"

#ifndef TEST_SERVICE_BACKEND_MODULE_PATH
#define TEST_SERVICE_BACKEND_MODULE_PATH "build/test_modules/libglm52_service_backend_module.so"
#endif

static void SparkTestServiceBackendLoadsRequiredInterface(void)
{
	SparkServiceBackendDynamicLibrary library;
	SparkServiceBackendConfiguration configuration;
	SparkServiceBackendView view;
	void *backend_state;

	memset(&library,0,sizeof(library));
	assert(SparkServiceBackendLoadInterfaceFromSharedObject(
		TEST_SERVICE_BACKEND_MODULE_PATH,
		SPARK_SERVICE_BACKEND_REQUIRED_PRODUCTION_CAPS,
		&library) == SPARK_STATUS_OK);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_SERVICE_BACKEND_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_SERVICE_BACKEND_CONFIGURATION_BYTES;
	backend_state = 0;
	assert(library.backend_interface.initialize(
		&configuration,
		&backend_state) == SPARK_STATUS_OK);
	assert(backend_state != 0);
	memset(&view,0,sizeof(view));
	assert(library.backend_interface.get_view(backend_state,&view) ==
		SPARK_STATUS_OK);
	assert(view.service != 0);
	assert(view.runtime_initialized == 1u);
	assert(view.local_control_ready == 1u);
	library.backend_interface.destroy(backend_state);
	SparkServiceBackendUnloadInterface(&library);
}

int main(void)
{
	SparkTestServiceBackendLoadsRequiredInterface();
	return 0;
}

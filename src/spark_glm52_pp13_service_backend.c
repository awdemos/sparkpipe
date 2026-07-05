#define _POSIX_C_SOURCE 200112L

#include "sparkpipe/spark_glm52_service_backend.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_model_driver.h"

#define SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE 1024u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_PORT_BASE 52100u
#define SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_MAGIC 0x35454650u

typedef struct SparkGlm52Pp13ServiceBackendState
{
	SparkGlm52Pp13RuntimeRankPlan rank_plan;
	SparkGlm52Pp13RuntimeFinalEventRoute final_event_route;
	SparkHiddenTransportDynamicLibrary transport_library;
	SparkHiddenTransportSession *output_transport_session;
	SparkLoadedModelDriver loaded_driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkGlm52ResidentDecodeStageProductionRunner runner;
	SparkTokenizer tokenizer;
	int32_t final_event_listen_fd;
	int32_t final_event_socket_fd;
	uint64_t final_event_receive_count;
	uint64_t final_event_receive_error_count;
	uint32_t last_final_event_token_count;
	uint32_t last_final_event_token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
	uint32_t initialized;
	uint32_t tokenizer_ready;
	uint32_t rank0_runtime_ready;
	char first_blocker[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
} SparkGlm52Pp13ServiceBackendState;

typedef struct SparkGlm52Pp13ServiceBackendFinalEvent
{
	uint32_t magic;
	uint32_t descriptor_bytes;
	uint32_t status;
	uint32_t program_id;
	uint32_t driver_dispatch_slot;
	uint32_t accepted_token_count;
	uint32_t completion_flags;
	uint32_t token_count;
	uint32_t token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t service_time_ns;
} SparkGlm52Pp13ServiceBackendFinalEvent;

static SparkGlm52Pp13ServiceBackendState SparkGlm52Pp13ServiceBackendSingleton;

static void SparkGlm52Pp13ServiceBackendSetBlocker(
	SparkGlm52Pp13ServiceBackendState *state,
	const char *message)
{
	if (state == 0 || state->first_blocker[0] != '\0')
		return;
	if (message == 0)
		message = "unknown PP13 service backend blocker";
	snprintf(state->first_blocker,sizeof(state->first_blocker),"%s",message);
}

static SparkStatus SparkGlm52Pp13ServiceBackendRequireText(
	const char *value,
	SparkGlm52Pp13ServiceBackendState *state,
	const char *message)
{
	if (value == 0 || value[0] == '\0')
	{
		SparkGlm52Pp13ServiceBackendSetBlocker(state,message);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Pp13ServiceBackendMaxActive(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	if (configuration->max_active_sequence_count != 0u)
		return configuration->max_active_sequence_count;
	return SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE;
}

static uint32_t SparkGlm52Pp13ServiceBackendPortBase(
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	if (configuration->port_base != 0u)
		return configuration->port_base;
	return SPARK_GLM52_PP13_SERVICE_BACKEND_DEFAULT_PORT_BASE;
}

static SparkStatus SparkGlm52Pp13ServiceBackendLoadTokenizer(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	SparkTokenizerCompiledFileConfiguration tokenizer_configuration;
	SparkStatus status;

	status = SparkGlm52Pp13ServiceBackendRequireText(
		configuration->tokenizer_path,
		state,
		"compiled C tokenizer path is missing");
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&tokenizer_configuration,0,sizeof(tokenizer_configuration));
	tokenizer_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
	tokenizer_configuration.descriptor_bytes =
		SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
	tokenizer_configuration.compiled_tokenizer_path =
		configuration->tokenizer_path;
	status = SparkTokenizerLoadCompiledFile(
		&state->tokenizer,
		&tokenizer_configuration);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52Pp13ServiceBackendSetBlocker(
			state,
			"compiled C tokenizer failed to load");
		return status;
	}
	state->tokenizer_ready = 1u;
	return SPARK_STATUS_OK;
}

static int32_t SparkGlm52Pp13ServiceBackendCreateListenSocket(
	const char *bind_address,
	uint32_t port)
{
	struct sockaddr_in address;
	int32_t fd;
	int32_t option;

	fd = socket(AF_INET,SOCK_STREAM,0);
	if (fd < 0)
		return -1;
	option = 1;
	if (setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&option,sizeof(option)) < 0)
	{
		close(fd);
		return -2;
	}
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)port);
	if (inet_pton(AF_INET,bind_address,&address.sin_addr) != 1)
	{
		close(fd);
		return -3;
	}
	if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0)
	{
		close(fd);
		return -4;
	}
	if (listen(fd,64) < 0)
	{
		close(fd);
		return -5;
	}
	return fd;
}

static int32_t SparkGlm52Pp13ServiceBackendSetNonblocking(int32_t fd)
{
	int32_t flags;

	flags = fcntl(fd,F_GETFL,0);
	if (flags < 0)
		return -1;
	if (fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0)
		return -2;
	return 0;
}

static SparkStatus SparkGlm52Pp13ServiceBackendLoadDriver(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration,
	char *error_buffer,
	uint32_t error_buffer_bytes)
{
	SparkModelDriverCreateRequest create_request;
	SparkStatus status;

	status = SparkLoadModelDriver(
		configuration->driver_shared_object_path,
		configuration->node_target,
		&state->loaded_driver,
		error_buffer,
		error_buffer_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	state->program = SparkFindLoadedModelDriverProgram(
		&state->loaded_driver,
		configuration->driver_program_name);
	if (state->program == 0)
		return SPARK_STATUS_NOT_FOUND;
	memset(&create_request,0,sizeof(create_request));
	create_request.node_id = state->rank_plan.host_name;
	create_request.node_target = configuration->node_target;
	create_request.node_context = 0;
	status = state->loaded_driver.interface->create(
		&create_request,
		&state->driver_instance);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->driver_instance == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendRank0Fail(
	SparkGlm52Pp13ServiceBackendState *state,
	SparkStatus status,
	const char *error_buffer,
	const char *message)
{
	if (error_buffer != 0 && error_buffer[0] != '\0')
		SparkGlm52Pp13ServiceBackendSetBlocker(state,error_buffer);
	else
		SparkGlm52Pp13ServiceBackendSetBlocker(state,message);
	return status;
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeRunner(
	SparkGlm52Pp13ServiceBackendState *state)
{
	SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;

	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
	configuration.flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
	configuration.driver_interface = state->loaded_driver.interface;
	configuration.driver_instance = state->driver_instance;
	configuration.program = state->program;
	configuration.execution_stream = 0;
	return SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
		&state->runner,
		&configuration);
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitializeRank0(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52ServiceBackendConfiguration *configuration)
{
	char error_buffer[SPARK_GLM52_SERVICE_BACKEND_BLOCKER_BYTES];
	SparkStatus status;

	error_buffer[0] = '\0';
	status = SparkGlm52Pp13RuntimeBuildRankPlan(
		0u,
		SparkGlm52Pp13ServiceBackendMaxActive(configuration),
		SparkGlm52Pp13ServiceBackendPortBase(configuration),
		&state->rank_plan,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build PP13 rank0 plan");
	status = SparkGlm52Pp13RuntimeBuildFinalEventRoute(
		SparkGlm52Pp13ServiceBackendPortBase(configuration),
		&state->final_event_route,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to build final event route");
	status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
		&state->rank_plan,
		configuration->fp8_pack_root,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to validate rank0 FP8 packs");
	status = SparkHiddenTransportLoadInterfaceFromSharedObject(
		configuration->transport_shared_object_path,
		SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
		&state->transport_library);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to load production transport");
	status = SparkHiddenTransportOpen(
		&state->rank_plan.output_endpoint,
		&state->transport_library.transport_interface,
		SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
		&state->output_transport_session);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to open rank0 output transport");
	status = SparkGlm52Pp13ServiceBackendLoadDriver(
		state,
		configuration,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to load GLM52 driver");
	status = SparkGlm52Pp13ServiceBackendInitializeRunner(state);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,status,error_buffer,"failed to initialize rank0 runner");
	state->final_event_listen_fd =
		SparkGlm52Pp13ServiceBackendCreateListenSocket(
			configuration->final_event_bind_address,
			state->final_event_route.listen_port);
	if (state->final_event_listen_fd < 0)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,
			SPARK_STATUS_ROUTE_NOT_FOUND,
			error_buffer,
			"failed to open rank0 final-event listener");
	if (SparkGlm52Pp13ServiceBackendSetNonblocking(
			state->final_event_listen_fd) < 0)
		return SparkGlm52Pp13ServiceBackendRank0Fail(
			state,
			SPARK_STATUS_INTERNAL_ERROR,
			error_buffer,
			"failed to make final-event listener nonblocking");
	state->rank0_runtime_ready = 1u;
	SparkGlm52Pp13ServiceBackendSetBlocker(
		state,
		"service submit path is not yet mapped to PP13 rank0 runner requests");
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13ServiceBackendInitialize(
	const SparkGlm52ServiceBackendConfiguration *configuration,
	void **backend_state)
{
	SparkGlm52Pp13ServiceBackendState *state;
	SparkStatus status;

	if (configuration == 0 || backend_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (configuration->abi_version != SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	state = &SparkGlm52Pp13ServiceBackendSingleton;
	memset(state,0,sizeof(*state));
	state->final_event_listen_fd = -1;
	state->final_event_socket_fd = -1;
	SparkLoadedModelDriverReset(&state->loaded_driver);
	SparkTokenizerReset(&state->tokenizer);
	state->initialized = 1u;
	*backend_state = state;
	status = SparkGlm52Pp13ServiceBackendRequireText(
		configuration->fp8_pack_root,
		state,
		"FP8 pack root is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->transport_shared_object_path,
			state,
			"production hidden transport shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->driver_shared_object_path,
			state,
			"GLM52 driver shared object is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->driver_program_name,
			state,
			"GLM52 driver program name is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->final_event_bind_address,
			state,
			"final-event bind address is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendRequireText(
			configuration->final_event_return_host,
			state,
			"final-event return host is missing");
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13ServiceBackendLoadTokenizer(
			state,
			configuration);
	if (status == SPARK_STATUS_OK)
		(void)SparkGlm52Pp13ServiceBackendInitializeRank0(
			state,
			configuration);
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13ServiceBackendDestroy(void *backend_state)
{
	SparkGlm52Pp13ServiceBackendState *state;

	state = (SparkGlm52Pp13ServiceBackendState *)backend_state;
	if (state == 0)
		return;
	if (state->final_event_socket_fd >= 0)
		close(state->final_event_socket_fd);
	if (state->final_event_listen_fd >= 0)
		close(state->final_event_listen_fd);
	if (state->loaded_driver.interface != 0 &&
		state->loaded_driver.interface->destroy != 0 &&
		state->driver_instance != 0)
		state->loaded_driver.interface->destroy(state->driver_instance);
	SparkUnloadModelDriver(&state->loaded_driver);
	SparkHiddenTransportClose(state->output_transport_session);
	SparkHiddenTransportUnloadInterface(&state->transport_library);
	SparkTokenizerDestroy(&state->tokenizer);
	memset(state,0,sizeof(*state));
}

static SparkStatus SparkGlm52Pp13ServiceBackendGetView(
	void *backend_state,
	SparkGlm52ServiceBackendView *view)
{
	SparkGlm52Pp13ServiceBackendState *state;

	state = (SparkGlm52Pp13ServiceBackendState *)backend_state;
	if (state == 0 || view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES;
	view->backend_ready = state->initialized != 0u ? 1u : 0u;
	view->pp13_ready = 0u;
	view->max_context_tokens = SPARK_GLM52_SERVING_DEFAULT_MAX_CONTEXT_TOKENS;
	view->production_contract_flags =
		SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
	view->service = 0;
	view->tokenizer = state->tokenizer_ready != 0u ? &state->tokenizer : 0;
	view->first_blocker = state->first_blocker;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13ServiceBackendAcceptFinalEventSocket(
	SparkGlm52Pp13ServiceBackendState *state)
{
	int32_t fd;

	if (state->final_event_listen_fd < 0 ||
		state->final_event_socket_fd >= 0)
		return;
	fd = accept(state->final_event_listen_fd,0,0);
	if (fd < 0)
	{
		if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
			state->final_event_receive_error_count += 1u;
		return;
	}
	if (SparkGlm52Pp13ServiceBackendSetNonblocking(fd) < 0)
	{
		close(fd);
		state->final_event_receive_error_count += 1u;
		return;
	}
	state->final_event_socket_fd = fd;
}

static void SparkGlm52Pp13ServiceBackendRecordFinalEvent(
	SparkGlm52Pp13ServiceBackendState *state,
	const SparkGlm52Pp13ServiceBackendFinalEvent *event)
{
	uint32_t token_count;

	if (event->magic != SPARK_GLM52_PP13_SERVICE_BACKEND_FINAL_EVENT_MAGIC ||
		event->descriptor_bytes != (uint32_t)sizeof(*event))
	{
		state->final_event_receive_error_count += 1u;
		return;
	}
	if ((event->completion_flags & SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS) == 0u)
	{
		state->final_event_receive_count += 1u;
		state->last_final_event_token_count = 0u;
		return;
	}
	token_count = event->token_count;
	if (token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
		token_count = SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
	memcpy(state->last_final_event_token_ids,event->token_ids,
		token_count * sizeof(state->last_final_event_token_ids[0u]));
	state->last_final_event_token_count = token_count;
	state->final_event_receive_count += 1u;
}

static SparkStatus SparkGlm52Pp13ServiceBackendPump(
	void *backend_state,
	uint32_t max_dispatch_steps,
	SparkGlm52ServiceStats *stats_out)
{
	SparkGlm52Pp13ServiceBackendState *state;
	uint8_t buffer[sizeof(SparkGlm52Pp13ServiceBackendFinalEvent)];
	ssize_t got;

	state = (SparkGlm52Pp13ServiceBackendState *)backend_state;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52Pp13ServiceBackendAcceptFinalEventSocket(state);
	if (state->final_event_socket_fd >= 0)
	{
		got = read(state->final_event_socket_fd,buffer,sizeof(buffer));
		if (got == (ssize_t)sizeof(buffer))
			SparkGlm52Pp13ServiceBackendRecordFinalEvent(
				state,
				(const SparkGlm52Pp13ServiceBackendFinalEvent *)buffer);
		else if (got < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
			errno != EINTR)
			state->final_event_receive_error_count += 1u;
	}
	if (stats_out != 0)
		memset(stats_out,0,sizeof(*stats_out));
	(void)max_dispatch_steps;
	return SPARK_STATUS_OK;
}

static const SparkGlm52ServiceBackendInterface SparkGlm52Pp13ServiceBackendInterface =
{
	SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION,
	SPARK_GLM52_SERVICE_BACKEND_INTERFACE_BYTES,
	SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_PP13_RUNTIME |
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_TOKENIZER,
	0u,
	SparkGlm52Pp13ServiceBackendInitialize,
	SparkGlm52Pp13ServiceBackendDestroy,
	SparkGlm52Pp13ServiceBackendGetView,
	SparkGlm52Pp13ServiceBackendPump
};

const SparkGlm52ServiceBackendInterface *SparkGlm52ServiceBackendGetInterface(void)
{
	return &SparkGlm52Pp13ServiceBackendInterface;
}

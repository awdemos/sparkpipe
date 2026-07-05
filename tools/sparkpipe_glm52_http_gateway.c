#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_service_backend.h"
#include "sparkpipe/spark_glm52_http_gateway.h"

#define SPARK_GLM52_GATEWAY_REQUEST_BYTES (SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_UPLOAD_BYTES + (1024u * 1024u))
#define SPARK_GLM52_GATEWAY_RESPONSE_BYTES (128u * 1024u)
#define SPARK_GLM52_GATEWAY_COMPAT_TEXT_BYTES SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_UPLOAD_BYTES
#define SPARK_GLM52_GATEWAY_DEFAULT_PORT 8080u
#define SPARK_GLM52_GATEWAY_DEFAULT_PUMP_STEPS 256u

typedef struct SparkGlm52GatewayConfig
{
	const char *bind_address;
	const char *api_key;
	const char *api_key_file;
	const char *service_backend_path;
	const char *fp8_pack_root;
	const char *transport_shared_object_path;
	const char *driver_shared_object_path;
	const char *node_context_builder_shared_object_path;
	const char *embedding_pack_path;
	const char *driver_program_name;
	const char *node_target;
	const char *tokenizer_path;
	const char *final_event_bind_address;
	const char *final_event_return_host;
	char api_key_storage[256];
	uint32_t port;
	uint32_t require_service_backend;
	uint32_t pump_steps;
	uint32_t max_active_sequence_count;
	uint32_t port_base;
} SparkGlm52GatewayConfig;

typedef struct SparkGlm52GatewayRuntime
{
	SparkGlm52GatewayConfig configuration;
	SparkGlm52ServiceBackendDynamicLibrary service_backend_library;
	void *service_backend_state;
	SparkGlm52ServiceBackendView service_backend_view;
	SparkGlm52ServiceClientId service_client_id;
	uint64_t next_client_request_id;
	uint32_t service_backend_attached;
} SparkGlm52GatewayRuntime;

static void SparkGlm52GatewayInitializeConfig(
	SparkGlm52GatewayConfig *configuration)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->bind_address = "127.0.0.1";
	configuration->driver_program_name = "glm52.pp13.rank.production";
	configuration->final_event_bind_address = "0.0.0.0";
	configuration->final_event_return_host = "spark0";
	configuration->port = SPARK_GLM52_GATEWAY_DEFAULT_PORT;
	configuration->pump_steps = SPARK_GLM52_GATEWAY_DEFAULT_PUMP_STEPS;
	configuration->max_active_sequence_count = 1024u;
	configuration->port_base = 52100u;
}

static int32_t SparkGlm52GatewayParseU32(const char *text,uint32_t *value_out)
{
	uint64_t value;
	uint32_t index;

	if (text == 0 || text[0] == '\0' || value_out == 0)
		return -1;
	value = 0u;
	for (index = 0u; text[index] != '\0'; ++index)
	{
		if (text[index] < '0' || text[index] > '9')
			return -2;
		value = ((value * 10u) + (uint32_t)(text[index] - '0'));
		if (value > 0xffffffffull)
			return -3;
	}
	*value_out = (uint32_t)value;
	return 0;
}

static int32_t SparkGlm52GatewayApplyArgument(
	SparkGlm52GatewayConfig *configuration,
	int argc,
	char **argv,
	int32_t *index)
{
	uint32_t parsed;

	if (strcmp(argv[*index],"--bind") == 0)
	{
		if ((*index + 1) >= argc)
			return -1;
		configuration->bind_address = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--port") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -2;
		configuration->port = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--api-key") == 0)
	{
		if ((*index + 1) >= argc)
			return -3;
		configuration->api_key = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--api-key-file") == 0)
	{
		if ((*index + 1) >= argc)
			return -4;
		configuration->api_key_file = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--service-backend-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -5;
		configuration->service_backend_path = argv[*index + 1];
		configuration->require_service_backend = 1u;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--require-service-backend") == 0)
	{
		configuration->require_service_backend = 1u;
		return 0;
	}
	if (strcmp(argv[*index],"--pump-steps") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -6;
		configuration->pump_steps = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--fp8-pack-root") == 0)
	{
		if ((*index + 1) >= argc)
			return -7;
		configuration->fp8_pack_root = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--transport-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -8;
		configuration->transport_shared_object_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--driver-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -9;
		configuration->driver_shared_object_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--node-context-builder-so") == 0)
	{
		if ((*index + 1) >= argc)
			return -10;
		configuration->node_context_builder_shared_object_path =
			argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--embedding-pack") == 0)
	{
		if ((*index + 1) >= argc)
			return -11;
		configuration->embedding_pack_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--program") == 0)
	{
		if ((*index + 1) >= argc)
			return -12;
		configuration->driver_program_name = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--node-target") == 0)
	{
		if ((*index + 1) >= argc)
			return -13;
		configuration->node_target = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--tokenizer") == 0)
	{
		if ((*index + 1) >= argc)
			return -14;
		configuration->tokenizer_path = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--max-active") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -15;
		configuration->max_active_sequence_count = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--port-base") == 0)
	{
		if ((*index + 1) >= argc ||
			SparkGlm52GatewayParseU32(argv[*index + 1],&parsed) < 0)
			return -16;
		configuration->port_base = parsed;
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--final-event-bind") == 0)
	{
		if ((*index + 1) >= argc)
			return -17;
		configuration->final_event_bind_address = argv[*index + 1];
		*index += 1;
		return 0;
	}
	if (strcmp(argv[*index],"--final-event-return-host") == 0)
	{
		if ((*index + 1) >= argc)
			return -18;
		configuration->final_event_return_host = argv[*index + 1];
		*index += 1;
		return 0;
	}
	return -19;
}

static int32_t SparkGlm52GatewayLoadApiKeyFile(
	SparkGlm52GatewayConfig *configuration)
{
	FILE *file;
	uint32_t bytes;

	if (configuration->api_key_file == 0)
		return 0;
	file = fopen(configuration->api_key_file,"rb");
	if (file == 0)
		return -1;
	bytes = (uint32_t)fread(
		configuration->api_key_storage,
		1u,
		sizeof(configuration->api_key_storage) - 1u,
		file);
	fclose(file);
	while (bytes != 0u &&
		(configuration->api_key_storage[bytes - 1u] == '\n' ||
		 configuration->api_key_storage[bytes - 1u] == '\r' ||
		 configuration->api_key_storage[bytes - 1u] == ' ' ||
		 configuration->api_key_storage[bytes - 1u] == '\t'))
		--bytes;
	configuration->api_key_storage[bytes] = '\0';
	if (bytes == 0u)
		return -2;
	configuration->api_key = configuration->api_key_storage;
	return 0;
}

static int32_t SparkGlm52GatewayParseArguments(
	SparkGlm52GatewayConfig *configuration,
	int argc,
	char **argv)
{
	int32_t index;

	for (index = 1; index < argc; ++index)
	{
		if (SparkGlm52GatewayApplyArgument(configuration,argc,argv,&index) < 0)
			return -1;
	}
	return 0;
}

static int32_t SparkGlm52GatewayRefreshBackendView(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkStatus status;

	if (runtime == 0 || runtime->service_backend_attached == 0u)
		return -1;
	status = runtime->service_backend_library.backend_interface.get_view(
		runtime->service_backend_state,
		&runtime->service_backend_view);
	if (status != SPARK_STATUS_OK)
		return -2;
	if (runtime->service_backend_view.abi_version !=
		SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION ||
		runtime->service_backend_view.descriptor_bytes !=
		SPARK_GLM52_SERVICE_BACKEND_VIEW_BYTES)
		return -3;
	return 0;
}

static int32_t SparkGlm52GatewayEnsureServiceClient(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkStatus status;

	if (runtime == 0 || runtime->service_backend_view.service == 0)
		return -1;
	if (runtime->service_client_id != 0u)
		return 0;
	status = SparkGlm52ServiceRegisterClient(
		runtime->service_backend_view.service,
		1u,
		&runtime->service_client_id);
	if (status != SPARK_STATUS_OK || runtime->service_client_id == 0u)
		return -2;
	runtime->next_client_request_id = 1u;
	return 0;
}

static int32_t SparkGlm52GatewayAttachServiceBackend(
	SparkGlm52GatewayRuntime *runtime)
{
	SparkGlm52ServiceBackendConfiguration backend_configuration;
	SparkStatus status;

	if (runtime == 0)
		return -1;
	if (runtime->configuration.service_backend_path == 0)
		return runtime->configuration.require_service_backend != 0u ? -2 : 0;
	status = SparkGlm52ServiceBackendLoadInterfaceFromSharedObject(
		runtime->configuration.service_backend_path,
		SPARK_GLM52_SERVICE_BACKEND_CAPABILITY_PP13_RUNTIME,
		&runtime->service_backend_library);
	if (status != SPARK_STATUS_OK)
		return -3;
	memset(&backend_configuration,0,sizeof(backend_configuration));
	backend_configuration.abi_version =
		SPARK_GLM52_SERVICE_BACKEND_ABI_VERSION;
	backend_configuration.descriptor_bytes =
		SPARK_GLM52_SERVICE_BACKEND_CONFIGURATION_BYTES;
	backend_configuration.max_active_sequence_count =
		runtime->configuration.max_active_sequence_count;
	backend_configuration.port_base = runtime->configuration.port_base;
	backend_configuration.fp8_pack_root = runtime->configuration.fp8_pack_root;
	backend_configuration.transport_shared_object_path =
		runtime->configuration.transport_shared_object_path;
	backend_configuration.driver_shared_object_path =
		runtime->configuration.driver_shared_object_path;
	backend_configuration.node_context_builder_shared_object_path =
		runtime->configuration.node_context_builder_shared_object_path;
	backend_configuration.embedding_pack_path =
		runtime->configuration.embedding_pack_path;
	backend_configuration.driver_program_name =
		runtime->configuration.driver_program_name;
	backend_configuration.node_target = runtime->configuration.node_target;
	backend_configuration.tokenizer_path = runtime->configuration.tokenizer_path;
	backend_configuration.final_event_bind_address =
		runtime->configuration.final_event_bind_address;
	backend_configuration.final_event_return_host =
		runtime->configuration.final_event_return_host;
	status = runtime->service_backend_library.backend_interface.initialize(
		&backend_configuration,
		&runtime->service_backend_state);
	if (status != SPARK_STATUS_OK || runtime->service_backend_state == 0)
		return -4;
	runtime->service_backend_attached = 1u;
	if (SparkGlm52GatewayRefreshBackendView(runtime) < 0)
		return -5;
	if (runtime->service_backend_view.service != 0 &&
		SparkGlm52GatewayEnsureServiceClient(runtime) < 0)
		return -6;
	return 0;
}

static void SparkGlm52GatewayDestroyServiceBackend(
	SparkGlm52GatewayRuntime *runtime)
{
	if (runtime == 0)
		return;
	if (runtime->service_backend_state != 0 &&
		runtime->service_backend_library.backend_interface.destroy != 0)
		runtime->service_backend_library.backend_interface.destroy(
			runtime->service_backend_state);
	runtime->service_backend_state = 0;
	runtime->service_backend_attached = 0u;
	SparkGlm52ServiceBackendUnloadInterface(&runtime->service_backend_library);
}

static int32_t SparkGlm52GatewayCreateListenSocket(
	const SparkGlm52GatewayConfig *configuration)
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
	address.sin_port = htons((uint16_t)configuration->port);
	if (inet_pton(AF_INET,configuration->bind_address,&address.sin_addr) != 1)
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

static char *SparkGlm52GatewayFindHeaderEnd(char *buffer,uint32_t bytes)
{
	uint32_t index;

	if (bytes < 4u)
		return 0;
	for (index = 0u; (index + 4u) <= bytes; ++index)
	{
		if (buffer[index] == '\r' && buffer[index + 1u] == '\n' &&
			buffer[index + 2u] == '\r' && buffer[index + 3u] == '\n')
			return &buffer[index];
	}
	return 0;
}

static const char *SparkGlm52GatewaySkipSpaces(const char *text)
{
	while (text != 0 && (*text == ' ' || *text == '\t'))
		++text;
	return text;
}

static void SparkGlm52GatewayExtractHeaders(
	char *headers,
	SparkGlm52HttpGatewayRequest *request,
	uint32_t *content_length_out)
{
	char *line;
	char *next;
	uint32_t value;

	*content_length_out = 0u;
	line = headers;
	while (line != 0 && line[0] != '\0')
	{
		next = strstr(line,"\r\n");
		if (next != 0)
		{
			next[0] = '\0';
			next += 2;
		}
		if (strncmp(line,"Content-Length:",15) == 0)
		{
			if (SparkGlm52GatewayParseU32(
					SparkGlm52GatewaySkipSpaces(line + 15),
					&value) == 0)
				*content_length_out = value;
		}
		if (strncmp(line,"content-length:",15) == 0)
		{
			if (SparkGlm52GatewayParseU32(
					SparkGlm52GatewaySkipSpaces(line + 15),
					&value) == 0)
				*content_length_out = value;
		}
		if (strncmp(line,"Authorization:",14) == 0)
		{
			request->authorization = SparkGlm52GatewaySkipSpaces(line + 14);
			request->authorization_bytes = (uint32_t)strlen(request->authorization);
		}
		if (strncmp(line,"authorization:",14) == 0)
		{
			request->authorization = SparkGlm52GatewaySkipSpaces(line + 14);
			request->authorization_bytes = (uint32_t)strlen(request->authorization);
		}
		line = next;
	}
}

static int32_t SparkGlm52GatewayParseRequestLine(
	char *line,
	SparkGlm52HttpGatewayRequest *request)
{
	char *path;
	char *version;

	path = strchr(line,' ');
	if (path == 0)
		return -1;
	path[0] = '\0';
	++path;
	version = strchr(path,' ');
	if (version == 0)
		return -2;
	version[0] = '\0';
	request->method = line;
	request->path = path;
	return 0;
}

static int32_t SparkGlm52GatewayReadRequest(
	int32_t fd,
	char *buffer,
	uint32_t capacity,
	SparkGlm52HttpGatewayRequest *request)
{
	char *header_end;
	char *body;
	char *line_end;
	uint32_t bytes;
	uint32_t header_bytes;
	uint32_t content_length;
	int32_t got;

	bytes = 0u;
	header_end = 0;
	while (bytes < capacity)
	{
		got = (int32_t)recv(fd,buffer + bytes,capacity - bytes,0);
		if (got <= 0)
			return -1;
		bytes += (uint32_t)got;
		header_end = SparkGlm52GatewayFindHeaderEnd(buffer,bytes);
		if (header_end != 0)
			break;
	}
	if (header_end == 0)
		return -2;
	header_bytes = (uint32_t)(header_end - buffer);
	body = header_end + 4;
	buffer[header_bytes] = '\0';
	line_end = strstr(buffer,"\r\n");
	if (line_end == 0)
		return -3;
	line_end[0] = '\0';
	if (SparkGlm52GatewayParseRequestLine(buffer,request) < 0)
		return -4;
	SparkGlm52GatewayExtractHeaders(line_end + 2,request,&content_length);
	if (content_length > (capacity - (uint32_t)(body - buffer)))
		return -6;
	while (((uint32_t)(bytes - (uint32_t)(body - buffer))) < content_length &&
		bytes < capacity)
	{
		got = (int32_t)recv(fd,buffer + bytes,capacity - bytes,0);
		if (got <= 0)
			return -5;
		bytes += (uint32_t)got;
	}
	request->body = body;
	request->body_bytes = content_length;
	return 0;
}

static const char *SparkGlm52GatewayStatusText(uint32_t status_code)
{
	if (status_code == 200u)
		return "OK";
	if (status_code == 202u)
		return "Accepted";
	if (status_code == 400u)
		return "Bad Request";
	if (status_code == 401u)
		return "Unauthorized";
	if (status_code == 404u)
		return "Not Found";
	if (status_code == 503u)
		return "Service Unavailable";
	return "Bad Request";
}

static int32_t SparkGlm52GatewaySendResponse(
	int32_t fd,
	const SparkGlm52HttpGatewayResponse *response)
{
	char header[1024];
	int32_t header_bytes;

	header_bytes = snprintf(
		header,
		sizeof(header),
		"HTTP/1.1 %u %s\r\n"
		"Content-Type: %s\r\n"
		"Content-Length: %u\r\n"
		"Access-Control-Allow-Origin: *\r\n"
		"Access-Control-Allow-Headers: authorization, content-type\r\n"
		"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
		"Cache-Control: no-cache\r\n"
		"Connection: close\r\n\r\n",
		response->status_code,
		SparkGlm52GatewayStatusText(response->status_code),
		response->content_type,
		response->body_bytes);
	if (header_bytes < 0 || (uint32_t)header_bytes >= sizeof(header))
		return -1;
	if (send(fd,header,(uint32_t)header_bytes,0) != header_bytes)
		return -2;
	if (send(fd,response->body,response->body_bytes,0) !=
		(int32_t)response->body_bytes)
		return -3;
	return 0;
}

static SparkStatus SparkGlm52GatewayBuildBadRequest(
	SparkGlm52HttpGatewayResponse *response)
{
	if (response == 0 || response->body == 0 || response->body_capacity < 64u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	response->status_code = 400u;
	response->flags = 0u;
	response->content_type = "application/json";
	strcpy(response->body,"{\"error\":{\"type\":\"bad_request\",\"message\":\"invalid or oversized request\"}}\n");
	response->body_bytes = (uint32_t)strlen(response->body);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52GatewayBuildResponse(
	SparkGlm52GatewayRuntime *runtime,
	const SparkGlm52HttpGatewayRequest *request,
	SparkGlm52HttpGatewayResponse *response)
{
	static char compat_text[SPARK_GLM52_GATEWAY_COMPAT_TEXT_BYTES];
	SparkGlm52CompatTextRequest compat_request;
	SparkGlm52ServiceSubmitResult submit_result;
	SparkGlm52ServiceStats service_stats;
	uint32_t route;
	uint32_t stream;
	SparkStatus status;

	route = SparkGlm52HttpGatewayRoute(request);
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_CORS_PREFLIGHT)
		return SparkGlm52HttpGatewayBuildCorsPreflight(response);
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI)
		return SparkGlm52HttpGatewayBuildDemoUi(response);
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH)
	{
		memset(&service_stats,0,sizeof(service_stats));
		if (runtime->service_backend_attached != 0u &&
			SparkGlm52GatewayRefreshBackendView(runtime) == 0 &&
			runtime->service_backend_view.service != 0)
		{
			(void)SparkGlm52ServiceGetStats(
				runtime->service_backend_view.service,
				&service_stats);
		}
		return SparkGlm52HttpGatewayBuildServiceHealth(
			response,
			runtime->service_backend_attached != 0u ? &service_stats : 0,
			runtime->service_backend_view.backend_ready,
			runtime->service_backend_view.pp13_ready,
			runtime->service_backend_view.max_context_tokens != 0u ?
				runtime->service_backend_view.max_context_tokens :
				SPARK_GLM52_HTTP_GATEWAY_DEFAULT_MAX_CONTEXT_TOKENS,
			runtime->service_backend_view.production_contract_flags,
			runtime->service_backend_view.first_blocker);
	}
	if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE)
		return SparkGlm52HttpGatewayBuildNotFound(response);
	if (SparkGlm52HttpGatewayAuthorizationMatches(
			request,
			runtime->configuration.api_key) == 0u)
		return SparkGlm52HttpGatewayBuildUnauthorized(response);
	stream = SparkGlm52HttpGatewayBodyRequestsStream(
		request->body,
		request->body_bytes);
	if (runtime->service_backend_attached == 0u ||
		SparkGlm52GatewayRefreshBackendView(runtime) < 0 ||
		runtime->service_backend_view.service == 0 ||
		runtime->service_backend_view.backend_ready == 0u ||
		runtime->service_backend_view.pp13_ready == 0u ||
		SparkGlm52GatewayEnsureServiceClient(runtime) < 0)
		return SparkGlm52HttpGatewayBuildBackendUnavailable(response,stream);
	SparkGlm52CompatInitializeTextRequest(
		&compat_request,
		compat_text,
		sizeof(compat_text));
	memset(&submit_result,0,sizeof(submit_result));
	status = SparkGlm52HttpGatewaySubmitJsonToService(
		runtime->service_backend_view.service,
		request,
		runtime->service_client_id,
		runtime->next_client_request_id,
		&compat_request,
		&submit_result,
		response);
	if (status == SPARK_STATUS_OK)
	{
		runtime->next_client_request_id += 1u;
		if (runtime->next_client_request_id == 0u)
			runtime->next_client_request_id = 1u;
		(void)runtime->service_backend_library.backend_interface.pump(
			runtime->service_backend_state,
			runtime->configuration.pump_steps,
			&service_stats);
		return SPARK_STATUS_OK;
	}
	return SparkGlm52HttpGatewayBuildBackendUnavailable(response,stream);
}

static int32_t SparkGlm52GatewayServeOne(
	SparkGlm52GatewayRuntime *runtime,
	int32_t client_fd)
{
	static char request_buffer[SPARK_GLM52_GATEWAY_REQUEST_BYTES];
	static char response_buffer[SPARK_GLM52_GATEWAY_RESPONSE_BYTES];
	SparkGlm52HttpGatewayRequest request;
	SparkGlm52HttpGatewayResponse response;
	SparkStatus status;
	int32_t read_status;

	SparkGlm52HttpGatewayInitializeRequest(&request);
	SparkGlm52HttpGatewayInitializeResponse(
		&response,
		response_buffer,
		sizeof(response_buffer));
	read_status = SparkGlm52GatewayReadRequest(
		client_fd,
		request_buffer,
		sizeof(request_buffer),
		&request);
	if (read_status < 0)
	{
		if (read_status == -6)
			(void)SparkGlm52GatewayBuildBadRequest(&response);
		else
			(void)SparkGlm52HttpGatewayBuildNotFound(&response);
	}
	else
	{
		status = SparkGlm52GatewayBuildResponse(runtime,&request,&response);
		if (status != SPARK_STATUS_OK)
			(void)SparkGlm52HttpGatewayBuildBackendUnavailable(&response,0u);
	}
	return SparkGlm52GatewaySendResponse(client_fd,&response);
}

int main(int argc,char **argv)
{
	SparkGlm52GatewayRuntime runtime;
	int32_t listen_fd;
	int32_t client_fd;

	(void)signal(SIGPIPE,SIG_IGN);
	memset(&runtime,0,sizeof(runtime));
	SparkGlm52GatewayInitializeConfig(&runtime.configuration);
	if (SparkGlm52GatewayParseArguments(&runtime.configuration,argc,argv) < 0)
	{
		fprintf(stderr,"usage: %s [--bind ip] [--port n] [--api-key key] [--api-key-file path] [--service-backend-so path] [--require-service-backend] [--pump-steps n] [--fp8-pack-root dir] [--transport-so path] [--driver-so path] [--node-context-builder-so path] [--embedding-pack path] [--tokenizer path]\n",argv[0]);
		return 2;
	}
	if (SparkGlm52GatewayLoadApiKeyFile(&runtime.configuration) < 0)
	{
		fprintf(stderr,"api key file failed\n");
		return 2;
	}
	if (SparkGlm52GatewayAttachServiceBackend(&runtime) < 0)
	{
		fprintf(stderr,"service backend attach failed\n");
		SparkGlm52GatewayDestroyServiceBackend(&runtime);
		return 2;
	}
	listen_fd = SparkGlm52GatewayCreateListenSocket(&runtime.configuration);
	if (listen_fd < 0)
	{
		fprintf(stderr,"listen failed: %s\n",strerror(errno));
		SparkGlm52GatewayDestroyServiceBackend(&runtime);
		return 3;
	}
	fprintf(stderr,"sparkpipe glm52 gateway listening on %s:%u\n",
		runtime.configuration.bind_address,
		runtime.configuration.port);
	for (;;)
	{
		client_fd = accept(listen_fd,0,0);
		if (client_fd < 0)
			continue;
		(void)SparkGlm52GatewayServeOne(&runtime,client_fd);
		close(client_fd);
	}
	SparkGlm52GatewayDestroyServiceBackend(&runtime);
	return 0;
}

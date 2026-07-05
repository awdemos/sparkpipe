#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_HTTP_GATEWAY_ABI_VERSION 1u
#define SPARK_GLM52_HTTP_GATEWAY_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkGlm52HttpGatewayRequest))
#define SPARK_GLM52_HTTP_GATEWAY_RESPONSE_BYTES \
    ((uint32_t)sizeof(SparkGlm52HttpGatewayResponse))

#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE 0u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI 1u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH 2u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT 3u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS 4u
#define SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES 5u

#define SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM 0x00000001u

typedef struct SparkGlm52HttpGatewayRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *method;
    const char *path;
    const char *body;
    uint32_t body_bytes;
    const char *authorization;
    uint32_t authorization_bytes;
} SparkGlm52HttpGatewayRequest;

typedef struct SparkGlm52HttpGatewayResponse
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t status_code;
    uint32_t flags;
    const char *content_type;
    char *body;
    uint32_t body_capacity;
    uint32_t body_bytes;
} SparkGlm52HttpGatewayResponse;

void SparkGlm52HttpGatewayInitializeRequest(
    SparkGlm52HttpGatewayRequest *request);

void SparkGlm52HttpGatewayInitializeResponse(
    SparkGlm52HttpGatewayResponse *response,
    char *body,
    uint32_t body_capacity);

uint32_t SparkGlm52HttpGatewayRoute(
    const SparkGlm52HttpGatewayRequest *request);

SparkStatus SparkGlm52HttpGatewayBuildDemoUi(
    SparkGlm52HttpGatewayResponse *response);

SparkStatus SparkGlm52HttpGatewayBuildHealth(
    SparkGlm52HttpGatewayResponse *response,
    uint32_t backend_ready,
    uint32_t pp13_ready);

SparkStatus SparkGlm52HttpGatewayBuildBackendUnavailable(
    SparkGlm52HttpGatewayResponse *response,
    uint32_t stream);

SparkStatus SparkGlm52HttpGatewayBuildUnauthorized(
    SparkGlm52HttpGatewayResponse *response);

SparkStatus SparkGlm52HttpGatewayBuildNotFound(
    SparkGlm52HttpGatewayResponse *response);

uint32_t SparkGlm52HttpGatewayBodyRequestsStream(
    const char *body,
    uint32_t body_bytes);

uint32_t SparkGlm52HttpGatewayAuthorizationMatches(
    const SparkGlm52HttpGatewayRequest *request,
    const char *api_key);

#ifdef __cplusplus
}
#endif

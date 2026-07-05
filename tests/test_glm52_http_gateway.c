#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_http_gateway.h"

static void SparkTestHttpGatewayRoutes(void)
{
	SparkGlm52HttpGatewayRequest request;

	SparkGlm52HttpGatewayInitializeRequest(&request);
	request.method = "GET";
	request.path = "/";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI);
	request.path = "/health";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH);
	request.method = "POST";
	request.path = "/v1/chat/completions";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT);
	request.path = "/v1/completions";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS);
	request.path = "/v1/messages";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES);
	request.path = "/bad";
	assert(SparkGlm52HttpGatewayRoute(&request) ==
		SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE);
}

static void SparkTestHttpGatewayBuildsHealth(void)
{
	SparkGlm52HttpGatewayResponse response;
	char body[512];

	SparkGlm52HttpGatewayInitializeResponse(&response,body,sizeof(body));
	assert(SparkGlm52HttpGatewayBuildHealth(&response,1u,0u) ==
		SPARK_STATUS_OK);
	assert(response.status_code == 200u);
	assert(strcmp(response.content_type,"application/json") == 0);
	assert(strstr(body,"\"backend_ready\":1") != 0);
	assert(strstr(body,"\"pp13_ready\":0") != 0);
}

static void SparkTestHttpGatewayBuildsSseUnavailable(void)
{
	SparkGlm52HttpGatewayResponse response;
	char body[512];

	SparkGlm52HttpGatewayInitializeResponse(&response,body,sizeof(body));
	assert(SparkGlm52HttpGatewayBuildBackendUnavailable(&response,1u) ==
		SPARK_STATUS_OK);
	assert(response.status_code == 503u);
	assert((response.flags & SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM) != 0u);
	assert(strcmp(response.content_type,"text/event-stream") == 0);
	assert(strstr(body,"event: error") != 0);
}

static void SparkTestHttpGatewayAuth(void)
{
	SparkGlm52HttpGatewayRequest request;
	static const char Auth[] = "Bearer secret";

	SparkGlm52HttpGatewayInitializeRequest(&request);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,0) == 1u);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,"") == 1u);
	request.authorization = Auth;
	request.authorization_bytes = (uint32_t)strlen(Auth);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,"secret") == 1u);
	assert(SparkGlm52HttpGatewayAuthorizationMatches(&request,"bad") == 0u);
}

static void SparkTestHttpGatewayStreamFlag(void)
{
	static const char StreamBody[] = "{\"model\":\"glm-5.2\",\"stream\" : true}";
	static const char PlainBody[] = "{\"model\":\"glm-5.2\"}";

	assert(SparkGlm52HttpGatewayBodyRequestsStream(
		StreamBody,
		(uint32_t)strlen(StreamBody)) == 1u);
	assert(SparkGlm52HttpGatewayBodyRequestsStream(
		PlainBody,
		(uint32_t)strlen(PlainBody)) == 0u);
}

int main(void)
{
	SparkTestHttpGatewayRoutes();
	SparkTestHttpGatewayBuildsHealth();
	SparkTestHttpGatewayBuildsSseUnavailable();
	SparkTestHttpGatewayAuth();
	SparkTestHttpGatewayStreamFlag();
	return 0;
}

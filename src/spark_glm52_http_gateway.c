#include "sparkpipe/spark_glm52_http_gateway.h"

#include <stdio.h>
#include <string.h>

static uint32_t SparkGlm52HttpStringEquals(const char *left,const char *right)
{
	if (left == 0 || right == 0)
		return 0u;
	return strcmp(left,right) == 0;
}

static uint32_t SparkGlm52HttpBytesMatch(
	const char *left,
	uint32_t left_bytes,
	const char *right)
{
	uint32_t right_bytes;

	if (left == 0 || right == 0)
		return 0u;
	right_bytes = (uint32_t)strlen(right);
	if (left_bytes != right_bytes)
		return 0u;
	return memcmp(left,right,right_bytes) == 0;
}

static SparkStatus SparkGlm52HttpWriteBody(
	SparkGlm52HttpGatewayResponse *response,
	const char *content_type,
	uint32_t status_code,
	uint32_t flags,
	const char *body)
{
	int32_t written;

	if (response == 0 || response->body == 0 || body == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(response->body,response->body_capacity,"%s",body);
	if (written < 0)
		return SPARK_STATUS_INTERNAL_ERROR;
	if ((uint32_t)written >= response->body_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	response->status_code = status_code;
	response->flags = flags;
	response->content_type = content_type;
	response->body_bytes = (uint32_t)written;
	return SPARK_STATUS_OK;
}

void SparkGlm52HttpGatewayInitializeRequest(
	SparkGlm52HttpGatewayRequest *request)
{
	if (request == 0)
		return;
	memset(request,0,sizeof(*request));
	request->abi_version = SPARK_GLM52_HTTP_GATEWAY_ABI_VERSION;
	request->descriptor_bytes = SPARK_GLM52_HTTP_GATEWAY_REQUEST_BYTES;
}

void SparkGlm52HttpGatewayInitializeResponse(
	SparkGlm52HttpGatewayResponse *response,
	char *body,
	uint32_t body_capacity)
{
	if (response == 0)
		return;
	memset(response,0,sizeof(*response));
	response->abi_version = SPARK_GLM52_HTTP_GATEWAY_ABI_VERSION;
	response->descriptor_bytes = SPARK_GLM52_HTTP_GATEWAY_RESPONSE_BYTES;
	response->body = body;
	response->body_capacity = body_capacity;
}

uint32_t SparkGlm52HttpGatewayRoute(
	const SparkGlm52HttpGatewayRequest *request)
{
	if (request == 0 || request->method == 0 || request->path == 0)
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE;
	if (SparkGlm52HttpStringEquals(request->method,"GET") &&
		SparkGlm52HttpStringEquals(request->path,"/"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI;
	if (SparkGlm52HttpStringEquals(request->method,"GET") &&
		SparkGlm52HttpStringEquals(request->path,"/health"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH;
	if (SparkGlm52HttpStringEquals(request->method,"POST") &&
		SparkGlm52HttpStringEquals(request->path,"/v1/chat/completions"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT;
	if (SparkGlm52HttpStringEquals(request->method,"POST") &&
		SparkGlm52HttpStringEquals(request->path,"/v1/completions"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS;
	if (SparkGlm52HttpStringEquals(request->method,"POST") &&
		SparkGlm52HttpStringEquals(request->path,"/v1/messages"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES;
	return SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE;
}

SparkStatus SparkGlm52HttpGatewayBuildDemoUi(
	SparkGlm52HttpGatewayResponse *response)
{
	static const char Body[] =
		"<!doctype html><html><head><meta charset=\"utf-8\">"
		"<title>SparkPipe GLM 5.2</title>"
		"<style>body{font-family:system-ui;margin:2rem;max-width:900px}"
		"textarea{width:100%;height:14rem}pre{white-space:pre-wrap;background:#111;color:#eee;padding:1rem}"
		"input,button,textarea{font:inherit}button{padding:.55rem 1rem}</style></head>"
		"<body><h1>SparkPipe GLM 5.2</h1>"
		"<p>POSTs OpenAI-compatible JSON to <code>/v1/chat/completions</code> and displays the streaming response.</p>"
		"<p><input id=\"key\" placeholder=\"Bearer token\" type=\"password\" style=\"width:100%\"></p>"
		"<textarea id=\"prompt\">Explain this C code:\\n\\nint add(int a,int b){return a+b;}</textarea>"
		"<p><button id=\"run\">Send</button></p><pre id=\"out\"></pre>"
		"<script>"
		"document.getElementById('run').onclick=async()=>{"
		"const out=document.getElementById('out');out.textContent='';"
		"const key=document.getElementById('key').value;"
		"const body={model:'glm-5.2',stream:true,messages:[{role:'user',content:document.getElementById('prompt').value}]};"
		"const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json','authorization':'Bearer '+key},body:JSON.stringify(body)});"
		"const reader=r.body.getReader();const dec=new TextDecoder();"
		"for(;;){const x=await reader.read();if(x.done)break;out.textContent+=dec.decode(x.value);}"
		"};"
		"</script></body></html>";
	return SparkGlm52HttpWriteBody(response,"text/html; charset=utf-8",200u,0u,Body);
}

SparkStatus SparkGlm52HttpGatewayBuildHealth(
	SparkGlm52HttpGatewayResponse *response,
	uint32_t backend_ready,
	uint32_t pp13_ready)
{
	int32_t written;

	if (response == 0 || response->body == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(
		response->body,
		response->body_capacity,
		"{\"service\":\"sparkpipe-glm52\",\"backend_ready\":%u,\"pp13_ready\":%u}\n",
		backend_ready != 0u ? 1u : 0u,
		pp13_ready != 0u ? 1u : 0u);
	if (written < 0)
		return SPARK_STATUS_INTERNAL_ERROR;
	if ((uint32_t)written >= response->body_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	response->status_code = 200u;
	response->flags = 0u;
	response->content_type = "application/json";
	response->body_bytes = (uint32_t)written;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52HttpGatewayBuildBackendUnavailable(
	SparkGlm52HttpGatewayResponse *response,
	uint32_t stream)
{
	static const char JsonBody[] =
		"{\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 PP13 backend is not attached\"}}\n";
	static const char StreamBody[] =
		"event: error\n"
		"data: {\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 PP13 backend is not attached\"}}\n\n";

	if (stream != 0u)
		return SparkGlm52HttpWriteBody(
			response,
			"text/event-stream",
			503u,
			SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM,
			StreamBody);
	return SparkGlm52HttpWriteBody(response,"application/json",503u,0u,JsonBody);
}

SparkStatus SparkGlm52HttpGatewayBuildUnauthorized(
	SparkGlm52HttpGatewayResponse *response)
{
	return SparkGlm52HttpWriteBody(
		response,
		"application/json",
		401u,
		0u,
		"{\"error\":{\"type\":\"unauthorized\",\"message\":\"missing or invalid bearer token\"}}\n");
}

SparkStatus SparkGlm52HttpGatewayBuildNotFound(
	SparkGlm52HttpGatewayResponse *response)
{
	return SparkGlm52HttpWriteBody(
		response,
		"application/json",
		404u,
		0u,
		"{\"error\":{\"type\":\"not_found\",\"message\":\"unknown endpoint\"}}\n");
}

uint32_t SparkGlm52HttpGatewayBodyRequestsStream(
	const char *body,
	uint32_t body_bytes)
{
	uint32_t index;
	uint32_t cursor;

	if (body == 0 || body_bytes < 12u)
		return 0u;
	for (index = 0u; (index + 8u) <= body_bytes; ++index)
	{
		if (memcmp(&body[index],"\"stream\"",8u) != 0)
			continue;
		cursor = index + 8u;
		while (cursor < body_bytes &&
			(body[cursor] == ' ' || body[cursor] == '\t' ||
			 body[cursor] == '\r' || body[cursor] == '\n'))
			++cursor;
		if (cursor >= body_bytes || body[cursor] != ':')
			continue;
		++cursor;
		while (cursor < body_bytes &&
			(body[cursor] == ' ' || body[cursor] == '\t' ||
			 body[cursor] == '\r' || body[cursor] == '\n'))
			++cursor;
		if ((cursor + 4u) <= body_bytes &&
			memcmp(&body[cursor],"true",4u) == 0)
			return 1u;
	}
	return 0u;
}

uint32_t SparkGlm52HttpGatewayAuthorizationMatches(
	const SparkGlm52HttpGatewayRequest *request,
	const char *api_key)
{
	char expected[512];
	int32_t written;

	if (api_key == 0 || api_key[0] == '\0')
		return 1u;
	if (request == 0)
		return 0u;
	written = snprintf(expected,sizeof(expected),"Bearer %s",api_key);
	if (written < 0 || (uint32_t)written >= sizeof(expected))
		return 0u;
	return SparkGlm52HttpBytesMatch(
		request->authorization,
		request->authorization_bytes,
		expected);
}

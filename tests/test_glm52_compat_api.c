#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_chat_template.h"
#include "sparkpipe/spark_glm52_compat_api.h"

#define SPARK_TEST_GLM52_CHAT_BEGIN \
	"[gMASK]<sop><|system|>Reasoning Effort: Max"
#define SPARK_TEST_GLM52_CHAT_END "<|assistant|><think>"
#define SPARK_TEST_GLM52_NO_THINK_CHAT_BEGIN "[gMASK]<sop>"
#define SPARK_TEST_GLM52_NO_THINK_CHAT_END \
	"<|assistant|><think></think>"

static void SparkTestCompatOpenAiChat(void)
{
    static const char RequestJson[] =
        "{"
        "\"model\":\"glm-5.2\","
        "\"max_tokens\":17,"
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are terse.\"},"
        "{\"role\":\"user\",\"content\":\"Read this C code.\"}"
        "]"
        "}";
    SparkGlm52CompatTextRequest request;
    static const char Expected[] =
		SPARK_TEST_GLM52_CHAT_BEGIN
		"<|system|>You are terse."
		"<|user|>Read this C code."
		SPARK_TEST_GLM52_CHAT_END;
    char text[512];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    request.client_id = 10u;
    request.client_request_id = 20u;
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(request.output_token_budget == 17u);
	assert(request.text_bytes == (uint32_t)(sizeof(Expected) - 1u));
	assert(strcmp(text, Expected) == 0);
}

static void SparkTestCompatOpenAiPrompt(void)
{
    static const char RequestJson[] =
        "{\"model\":\"glm-5.2\",\"max_completion_tokens\":5,\"prompt\":\"plain prompt\"}";
    SparkGlm52CompatTextRequest request;
    char text[64];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(request.output_token_budget == 5u);
    assert(strcmp(text, "plain prompt") == 0);
}

static void SparkTestCompatOpenAiThinkingBudgets(void)
{
	static const char NoThinkJson[] =
		"{\"thinking_budget_tokens\":0,"
		"\"messages\":[{\"role\":\"user\",\"content\":\"Answer.\"}]}";
	static const char ThinkJson[] =
		"{\"thinking_token_budget\":1024,"
		"\"messages\":[{\"role\":\"user\",\"content\":\"Answer.\"}]}";
	static const char ConflictJson[] =
		"{\"thinking_budget_tokens\":0,\"thinking_token_budget\":1024,"
		"\"messages\":[{\"role\":\"user\",\"content\":\"Answer.\"}]}";
	SparkGlm52CompatTextRequest request;
	char text[256];

	SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
	assert(SparkGlm52CompatPrepareOpenAiJson(
		NoThinkJson,
		(uint32_t)strlen(NoThinkJson),
		&request) == SPARK_STATUS_OK);
	assert(request.thinking_token_budget == 0u);
	assert(strcmp(
		text,
		SPARK_TEST_GLM52_NO_THINK_CHAT_BEGIN
		"<|user|>Answer."
		SPARK_TEST_GLM52_NO_THINK_CHAT_END) == 0);
	SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
	assert(SparkGlm52CompatPrepareOpenAiJson(
		ThinkJson,
		(uint32_t)strlen(ThinkJson),
		&request) == SPARK_STATUS_OK);
	assert(request.thinking_token_budget == 1024u);
	assert(strcmp(
		text,
		SPARK_TEST_GLM52_CHAT_BEGIN
		"<|user|>Answer."
		SPARK_TEST_GLM52_CHAT_END) == 0);
	SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
	assert(SparkGlm52CompatPrepareOpenAiJson(
		ConflictJson,
		(uint32_t)strlen(ConflictJson),
		&request) == SPARK_STATUS_INVALID_ARGUMENT);
}


static void SparkTestCompatOpenAiChatWithFiles(void)
{
    static const char RequestJson[] =
        "{"
        "\"model\":\"glm-5.2\","
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Use the attachment.\"}]}],"
        "\"files\":[{\"filename\":\"notes.txt\",\"content\":\"alpha\\nbeta\"}]"
        "}";
    SparkGlm52CompatTextRequest request;
    char text[512];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
	assert(strstr(text, SPARK_TEST_GLM52_CHAT_BEGIN "<|user|>Use the attachment.") != 0);
    assert(strstr(text, "[uploaded file: notes.txt]") != 0);
    assert(strstr(text, "alpha\nbeta") != 0);
    assert(strstr(text, "[/uploaded file]") != 0);
	assert(strstr(text, SPARK_TEST_GLM52_CHAT_END) != 0);
}

static void SparkTestCompatAnthropicMessages(void)
{
    static const char RequestJson[] =
        "{"
        "\"model\":\"glm-5.2\","
        "\"system\":\"Stay exact.\","
        "\"max_tokens\":9,"
        "\"messages\":["
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"First\"},"
        "{\"type\":\"text\",\"text\":\" second\"}]},"
        "{\"role\":\"assistant\",\"content\":\"Ack\"}"
        "]"
        "}";
    SparkGlm52CompatTextRequest request;
    static const char Expected[] =
		SPARK_TEST_GLM52_CHAT_BEGIN
		"<|system|>Stay exact."
		"<|user|>First second"
		"<|assistant|><think></think>Ack"
		SPARK_TEST_GLM52_CHAT_END;
    char text[512];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareAnthropicJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(request.output_token_budget == 9u);
	assert(strcmp(text, Expected) == 0);
}

static void SparkTestCompatRejectsUnknownChatRole(void)
{
	static const char RequestJson[] =
		"{\"messages\":[{\"role\":\"alien\",\"content\":\"no\"}]}";
	SparkGlm52CompatTextRequest request;
	char text[256];

	SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
	assert(SparkGlm52CompatPrepareOpenAiJson(
		RequestJson,
		(uint32_t)strlen(RequestJson),
		&request) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestChatTemplateReasoningEffort(void)
{
	SparkGlm52ChatTemplateWriter writer;
	char text[128];

	assert(SparkGlm52ChatTemplateInitializeWriter(
		&writer,
		text,
		(uint32_t)sizeof(text),
		0u) == SPARK_STATUS_OK);
	assert(SparkGlm52ChatTemplateBegin(
		&writer,
		"high",
		SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING) == SPARK_STATUS_OK);
	assert(strcmp(
		text,
		"[gMASK]<sop><|system|>Reasoning Effort: High") == 0);
	assert(SparkGlm52ChatTemplateEndMessage(
		&writer,
		(SparkGlm52ChatTemplateRole)-1) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestCompatRejectsSmallBuffer(void)
{
    static const char RequestJson[] =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"too long for buffer\"}]}";
    SparkGlm52CompatTextRequest request;
    char text[8];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_CAPACITY_EXCEEDED);
}

int main(void)
{
    SparkTestCompatOpenAiChat();
    SparkTestCompatOpenAiPrompt();
	SparkTestCompatOpenAiThinkingBudgets();
    SparkTestCompatOpenAiChatWithFiles();
    SparkTestCompatAnthropicMessages();
	SparkTestCompatRejectsUnknownChatRole();
	SparkTestChatTemplateReasoningEffort();
    SparkTestCompatRejectsSmallBuffer();
    return 0;
}

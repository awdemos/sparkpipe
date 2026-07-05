#include "sparkpipe/spark_glm52_compat_api.h"

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_json.h"

static SparkStatus SparkGlm52CompatAppendBytes(
    SparkGlm52CompatTextRequest *request,
    const char *text,
    uint32_t text_bytes)
{
    if ( request == 0 || request->text == 0 || text == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( text_bytes > request->text_capacity ||
        request->text_bytes > (request->text_capacity - text_bytes) )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    memcpy(&request->text[request->text_bytes], text, text_bytes);
    request->text_bytes += text_bytes;
    if ( request->text_bytes < request->text_capacity )
        request->text[request->text_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendLiteral(
    SparkGlm52CompatTextRequest *request,
    const char *text)
{
    return SparkGlm52CompatAppendBytes(
        request,
        text,
        (uint32_t)strlen(text));
}

static SparkStatus SparkGlm52CompatAppendJsonString(
    const SparkJsonDocument *document,
    int32_t token_index,
    SparkGlm52CompatTextRequest *request)
{
    SparkStatus status;
    char *decoded_text;

    decoded_text = 0;
    status = SparkJsonCopyString(document, token_index, &decoded_text);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatAppendLiteral(request, decoded_text);
    free(decoded_text);
    return status;
}

static SparkStatus SparkGlm52CompatReadOptionalUInt32(
    const SparkJsonDocument *document,
    int32_t root_token_index,
    const char *member_name,
    uint32_t *value)
{
    int32_t member_token_index;

    member_token_index =
        SparkJsonFindObjectMember(document, root_token_index, member_name);
    if ( member_token_index < 0 )
        return SPARK_STATUS_OK;
    return SparkJsonGetUInt32(document, member_token_index, value);
}

static SparkStatus SparkGlm52CompatAppendContentToken(
    const SparkJsonDocument *document,
    int32_t content_token_index,
    SparkGlm52CompatTextRequest *request)
{
    uint32_t element_count;
    uint32_t element_index;
    SparkStatus status;

    if ( SparkJsonTokenIsType(document, content_token_index, SPARK_JSON_TOKEN_STRING) )
        return SparkGlm52CompatAppendJsonString(
            document,
            content_token_index,
            request);
    if ( !SparkJsonTokenIsType(document, content_token_index, SPARK_JSON_TOKEN_ARRAY) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    element_count = SparkJsonGetArrayElementCount(document, content_token_index);
    for (element_index=0u; element_index<element_count; element_index++)
    {
        int32_t element_token_index;
        int32_t text_token_index;

        element_token_index =
            SparkJsonGetArrayElement(document, content_token_index, element_index);
        if ( SparkJsonTokenIsType(document, element_token_index, SPARK_JSON_TOKEN_STRING) )
        {
            status = SparkGlm52CompatAppendJsonString(
                document,
                element_token_index,
                request);
            if ( status != SPARK_STATUS_OK )
                return status;
            continue;
        }
        if ( !SparkJsonTokenIsType(document, element_token_index, SPARK_JSON_TOKEN_OBJECT) )
            return SPARK_STATUS_INVALID_ARGUMENT;
        text_token_index =
            SparkJsonFindObjectMember(document, element_token_index, "text");
        if ( text_token_index >= 0 )
        {
            status = SparkGlm52CompatAppendJsonString(
                document,
                text_token_index,
                request);
            if ( status != SPARK_STATUS_OK )
                return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendRole(
    const SparkJsonDocument *document,
    int32_t message_token_index,
    SparkGlm52CompatTextRequest *request)
{
    int32_t role_token_index;
    SparkStatus status;

    role_token_index =
        SparkJsonFindObjectMember(document, message_token_index, "role");
    if ( role_token_index >= 0 )
        status = SparkGlm52CompatAppendJsonString(document, role_token_index, request);
    else
        status = SparkGlm52CompatAppendLiteral(request, "user");
    if ( status != SPARK_STATUS_OK )
        return status;
    return SparkGlm52CompatAppendLiteral(request, ": ");
}

static SparkStatus SparkGlm52CompatAppendMessages(
    const SparkJsonDocument *document,
    int32_t messages_token_index,
    SparkGlm52CompatTextRequest *request)
{
    uint32_t message_count;
    uint32_t message_index;
    SparkStatus status;

    if ( !SparkJsonTokenIsType(document, messages_token_index, SPARK_JSON_TOKEN_ARRAY) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    message_count = SparkJsonGetArrayElementCount(document, messages_token_index);
    if ( message_count == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (message_index=0u; message_index<message_count; message_index++)
    {
        int32_t message_token_index;
        int32_t content_token_index;

        message_token_index =
            SparkJsonGetArrayElement(document, messages_token_index, message_index);
        if ( !SparkJsonTokenIsType(document, message_token_index, SPARK_JSON_TOKEN_OBJECT) )
            return SPARK_STATUS_INVALID_ARGUMENT;
        content_token_index =
            SparkJsonFindObjectMember(document, message_token_index, "content");
        if ( content_token_index < 0 )
            return SPARK_STATUS_INVALID_ARGUMENT;
        status = SparkGlm52CompatAppendRole(document, message_token_index, request);
        if ( status != SPARK_STATUS_OK )
            return status;
        status = SparkGlm52CompatAppendContentToken(
            document,
            content_token_index,
            request);
        if ( status != SPARK_STATUS_OK )
            return status;
        status = SparkGlm52CompatAppendLiteral(request, "\n");
        if ( status != SPARK_STATUS_OK )
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatPrepareCommon(
    SparkJsonDocument *document,
    SparkGlm52CompatTextRequest *request)
{
    int32_t root_token_index;
    SparkStatus status;

    if ( request == 0 || request->abi_version != SPARK_GLM52_COMPAT_API_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_COMPAT_TEXT_REQUEST_BYTES ||
        request->text == 0 || request->text_capacity == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    root_token_index = SparkJsonGetRootToken(document);
    if ( !SparkJsonTokenIsType(document, root_token_index, SPARK_JSON_TOKEN_OBJECT) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    request->text_bytes = 0u;
    request->text[0] = '\0';
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "max_tokens",
        &request->output_token_budget);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "max_completion_tokens",
        &request->output_token_budget);
    if ( status != SPARK_STATUS_OK )
        return status;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatSubmitPrepared(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52CompatTextRequest *compat_request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkGlm52ServiceSubmitTextRequest request;

    SparkGlm52ServiceInitializeSubmitTextRequest(&request);
    request.flags = compat_request->flags;
    request.priority = compat_request->priority;
    request.thinking_token_budget = compat_request->thinking_token_budget;
    request.output_token_budget = compat_request->output_token_budget;
    request.max_prefill_tokens_per_step =
        compat_request->max_prefill_tokens_per_step;
    request.tokenizer_encode_flags = compat_request->tokenizer_encode_flags;
    request.client_id = compat_request->client_id;
    request.client_request_id = compat_request->client_request_id;
    request.sequence_id = compat_request->sequence_id;
    request.text = compat_request->text;
    request.text_bytes = compat_request->text_bytes;
    return SparkGlm52ServiceSubmitText(service, &request, result);
}

void SparkGlm52CompatInitializeTextRequest(
    SparkGlm52CompatTextRequest *request,
    char *text,
    uint32_t text_capacity)
{
    if ( request == 0 )
        return;
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_COMPAT_API_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_COMPAT_TEXT_REQUEST_BYTES;
    request->text = text;
    request->text_capacity = text_capacity;
    if ( text != 0 && text_capacity != 0u )
        text[0] = '\0';
}

SparkStatus SparkGlm52CompatPrepareOpenAiJson(
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t messages_token_index;
    int32_t prompt_token_index;
    SparkStatus status;

    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(json_text, json_bytes, &document);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatPrepareCommon(&document, request);
    if ( status == SPARK_STATUS_OK )
    {
        root_token_index = SparkJsonGetRootToken(&document);
        messages_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "messages");
        prompt_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "prompt");
        if ( messages_token_index >= 0 )
            status = SparkGlm52CompatAppendMessages(
                &document,
                messages_token_index,
                request);
        else if ( prompt_token_index >= 0 )
            status = SparkGlm52CompatAppendJsonString(
                &document,
                prompt_token_index,
                request);
        else
            status = SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkJsonDocumentDestroy(&document);
    return status;
}

SparkStatus SparkGlm52CompatPrepareAnthropicJson(
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t system_token_index;
    int32_t messages_token_index;
    SparkStatus status;

    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(json_text, json_bytes, &document);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatPrepareCommon(&document, request);
    if ( status == SPARK_STATUS_OK )
    {
        root_token_index = SparkJsonGetRootToken(&document);
        system_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "system");
        if ( system_token_index >= 0 )
        {
            status = SparkGlm52CompatAppendLiteral(request, "system: ");
            if ( status == SPARK_STATUS_OK )
                status = SparkGlm52CompatAppendJsonString(
                    &document,
                    system_token_index,
                    request);
            if ( status == SPARK_STATUS_OK )
                status = SparkGlm52CompatAppendLiteral(request, "\n");
        }
        messages_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "messages");
        if ( status == SPARK_STATUS_OK && messages_token_index >= 0 )
            status = SparkGlm52CompatAppendMessages(
                &document,
                messages_token_index,
                request);
        else if ( status == SPARK_STATUS_OK )
            status = SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkJsonDocumentDestroy(&document);
    return status;
}

SparkStatus SparkGlm52CompatSubmitOpenAiJson(
    SparkGlm52ServiceRuntime *service,
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkStatus status;

    status = SparkGlm52CompatPrepareOpenAiJson(json_text, json_bytes, request);
    if ( status != SPARK_STATUS_OK )
        return status;
    return SparkGlm52CompatSubmitPrepared(service, request, result);
}

SparkStatus SparkGlm52CompatSubmitAnthropicJson(
    SparkGlm52ServiceRuntime *service,
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkStatus status;

    status = SparkGlm52CompatPrepareAnthropicJson(json_text, json_bytes, request);
    if ( status != SPARK_STATUS_OK )
        return status;
    return SparkGlm52CompatSubmitPrepared(service, request, result);
}

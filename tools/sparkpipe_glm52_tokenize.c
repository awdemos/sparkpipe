#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tokenizer.h"

#include "sparkpipe/spark_glm52_model.h"

#define SPARK_GLM52_TOKENIZE_DEFAULT_TOKEN_CAPACITY \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_GLM52_TOKENIZE_CHAT_FLAG_ADD_GENERATION_PROMPT 0x00000001u
#define SPARK_GLM52_TOKENIZE_CHAT_FLAG_ENABLE_THINKING 0x00000002u

static const char SparkGlm52TokenizerFileName[] = "tokenizer.json";

typedef struct SparkGlm52TokenizeStringBuilder
{
    char *text;
    uint32_t text_bytes;
    uint32_t text_capacity;
} SparkGlm52TokenizeStringBuilder;

static int32_t SparkGlm52TokenizeParseU32(
    const char *text,
    uint32_t *value_out)
{
    uint64_t value;
    uint32_t index;

    if (text == 0 || text[0] == '\0' || value_out == 0)
    {
        return -1;
    }
    value = 0u;
    for (index = 0u; text[index] != '\0'; ++index)
    {
        if (text[index] < '0' || text[index] > '9')
        {
            return -2;
        }
        value = value * 10u + (uint32_t)(text[index] - '0');
        if (value > 0xffffffffull)
        {
            return -3;
        }
    }
    *value_out = (uint32_t)value;
    return 0;
}

static int32_t SparkGlm52TokenizeReadFile(
    const char *path,
    char **text_out,
    uint32_t *text_bytes_out)
{
    FILE *file;
    long file_bytes;
    char *text;
    size_t read_bytes;

    if (path == 0 || text_out == 0 || text_bytes_out == 0)
    {
        return -1;
    }
    *text_out = 0;
    *text_bytes_out = 0u;
    file = fopen(path, "rb");
    if (file == 0)
    {
        return -2;
    }
    if (fseek(file, 0, SEEK_END) != 0)
    {
        fclose(file);
        return -3;
    }
    file_bytes = ftell(file);
    if (file_bytes < 0 || (uint64_t)file_bytes > 0xffffffffull)
    {
        fclose(file);
        return -4;
    }
    if (fseek(file, 0, SEEK_SET) != 0)
    {
        fclose(file);
        return -5;
    }
    text = (char *)malloc((size_t)file_bytes + 1u);
    if (text == 0)
    {
        fclose(file);
        return -6;
    }
    read_bytes = fread(text, 1u, (size_t)file_bytes, file);
    if (read_bytes != (size_t)file_bytes)
    {
        free(text);
        fclose(file);
        return -7;
    }
    fclose(file);
    text[file_bytes] = '\0';
    *text_out = text;
    *text_bytes_out = (uint32_t)file_bytes;
    return 0;
}

static uint32_t SparkGlm52TokenizeStringLengthU32(
    const char *text)
{
    size_t text_bytes;

    if (text == 0)
    {
        return 0u;
    }
    text_bytes = strlen(text);
    if (text_bytes > 0xffffffffull)
    {
        return 0xffffffffu;
    }
    return (uint32_t)text_bytes;
}

static int32_t SparkGlm52TokenizeStringBuilderReserve(
    SparkGlm52TokenizeStringBuilder *builder,
    uint32_t needed_bytes)
{
    uint32_t resized_capacity;
    char *resized_text;

    if (builder == 0)
    {
        return -1;
    }
    if (needed_bytes <= builder->text_capacity)
    {
        return 0;
    }
    resized_capacity = builder->text_capacity == 0u ? 256u : builder->text_capacity;
    while (resized_capacity < needed_bytes)
    {
        if (resized_capacity > 0x80000000u)
        {
            return -2;
        }
        resized_capacity *= 2u;
    }
    resized_text = (char *)realloc(builder->text, (size_t)resized_capacity);
    if (resized_text == 0)
    {
        return -3;
    }
    builder->text = resized_text;
    builder->text_capacity = resized_capacity;
    return 0;
}

static int32_t SparkGlm52TokenizeStringBuilderAppend(
    SparkGlm52TokenizeStringBuilder *builder,
    const char *text,
    uint32_t text_bytes)
{
    if (builder == 0 || (text == 0 && text_bytes != 0u))
    {
        return -1;
    }
    if (builder->text_bytes > 0xffffffffu - text_bytes - 1u)
    {
        return -2;
    }
    if (SparkGlm52TokenizeStringBuilderReserve(
        builder,
        builder->text_bytes + text_bytes + 1u) != 0)
    {
        return -3;
    }
    if (text_bytes != 0u)
    {
        memcpy(builder->text + builder->text_bytes, text, text_bytes);
    }
    builder->text_bytes += text_bytes;
    builder->text[builder->text_bytes] = '\0';
    return 0;
}

static void SparkGlm52TokenizeStringBuilderDestroy(
    SparkGlm52TokenizeStringBuilder *builder)
{
    if (builder == 0)
    {
        return;
    }
    free(builder->text);
    memset(builder, 0, sizeof(*builder));
}

static int32_t SparkGlm52TokenizeBuildTokenizerPath(
    const char *model_dir,
    char **tokenizer_json_path_out)
{
    uint32_t model_dir_bytes;
    uint32_t needs_separator;
    uint32_t path_bytes;
    char *path;

    if (model_dir == 0 || tokenizer_json_path_out == 0)
    {
        return -1;
    }
    *tokenizer_json_path_out = 0;
    model_dir_bytes = SparkGlm52TokenizeStringLengthU32(model_dir);
    needs_separator = model_dir_bytes != 0u && model_dir[model_dir_bytes - 1u] != '/' ? 1u : 0u;
    if (model_dir_bytes >
        0xffffffffu - sizeof(SparkGlm52TokenizerFileName) - needs_separator)
    {
        return -2;
    }
    path_bytes = model_dir_bytes + needs_separator +
        sizeof(SparkGlm52TokenizerFileName) - 1u;
    path = (char *)malloc((size_t)path_bytes + 1u);
    if (path == 0)
    {
        return -3;
    }
    memcpy(path, model_dir, model_dir_bytes);
    if (needs_separator != 0u)
    {
        path[model_dir_bytes] = '/';
    }
    memcpy(
        path + model_dir_bytes + needs_separator,
        SparkGlm52TokenizerFileName,
        sizeof(SparkGlm52TokenizerFileName));
    *tokenizer_json_path_out = path;
    return 0;
}

static int32_t SparkGlm52TokenizeBuildSimpleChatPrompt(
    const char *prompt,
    uint32_t prompt_bytes,
    const char *system_prompt,
    uint32_t system_prompt_bytes,
    const char *reasoning_effort,
    uint32_t chat_flags,
    char **chat_text_out,
    uint32_t *chat_text_bytes_out)
{
    SparkGlm52TokenizeStringBuilder builder;
    uint32_t reasoning_effort_bytes;
    const char *thinking_open;

    if (prompt == 0 || chat_text_out == 0 || chat_text_bytes_out == 0)
    {
        return -1;
    }
    memset(&builder, 0, sizeof(builder));
    *chat_text_out = 0;
    *chat_text_bytes_out = 0u;
    if (reasoning_effort == 0)
    {
        reasoning_effort = "Max";
    }
    reasoning_effort_bytes = SparkGlm52TokenizeStringLengthU32(reasoning_effort);
    if (SparkGlm52TokenizeStringBuilderAppend(&builder, "[gMASK]<sop>", 12u) != 0)
    {
        SparkGlm52TokenizeStringBuilderDestroy(&builder);
        return -2;
    }
    if ((chat_flags & SPARK_GLM52_TOKENIZE_CHAT_FLAG_ENABLE_THINKING) != 0u)
    {
        if (SparkGlm52TokenizeStringBuilderAppend(&builder, "<|system|>Reasoning Effort: ", 29u) != 0 ||
            SparkGlm52TokenizeStringBuilderAppend(&builder, reasoning_effort, reasoning_effort_bytes) != 0)
        {
            SparkGlm52TokenizeStringBuilderDestroy(&builder);
            return -3;
        }
    }
    if (system_prompt != 0 && system_prompt_bytes != 0u)
    {
        if (SparkGlm52TokenizeStringBuilderAppend(&builder, "<|system|>", 10u) != 0 ||
            SparkGlm52TokenizeStringBuilderAppend(&builder, system_prompt, system_prompt_bytes) != 0)
        {
            SparkGlm52TokenizeStringBuilderDestroy(&builder);
            return -4;
        }
    }
    if (SparkGlm52TokenizeStringBuilderAppend(&builder, "<|user|>", 8u) != 0 ||
        SparkGlm52TokenizeStringBuilderAppend(&builder, prompt, prompt_bytes) != 0)
    {
        SparkGlm52TokenizeStringBuilderDestroy(&builder);
        return -5;
    }
    if ((chat_flags & SPARK_GLM52_TOKENIZE_CHAT_FLAG_ADD_GENERATION_PROMPT) != 0u)
    {
        thinking_open = (chat_flags & SPARK_GLM52_TOKENIZE_CHAT_FLAG_ENABLE_THINKING) != 0u ?
            "<think>" : "<think></think>";
        if (SparkGlm52TokenizeStringBuilderAppend(&builder, "<|assistant|>", 13u) != 0 ||
            SparkGlm52TokenizeStringBuilderAppend(
                &builder,
                thinking_open,
                SparkGlm52TokenizeStringLengthU32(thinking_open)) != 0)
        {
            SparkGlm52TokenizeStringBuilderDestroy(&builder);
            return -6;
        }
    }
    *chat_text_out = builder.text;
    *chat_text_bytes_out = builder.text_bytes;
    return 0;
}

static int32_t SparkGlm52TokenizeWriteTokens(
    FILE *file,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint32_t token_index;

    if (file == 0 || token_ids == 0)
    {
        return -1;
    }
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        if (fprintf(file, "%u\n", token_ids[token_index]) < 0)
        {
            return -2;
        }
    }
    return 0;
}

static int32_t SparkGlm52TokenizeUsage(
    const char *program)
{
    fprintf(stderr,
        "usage: %s (--model-dir dir | --tokenizer-json path) (--prompt text | --prompt-file path) [--tokens-out path] [--chat] [--system-prompt text] [--system-prompt-file path] [--no-add-generation-prompt] [--disable-thinking] [--reasoning-effort Max|High] [--capacity n]\n",
        program);
    return 2;
}

int main(
    int argc,
    char **argv)
{
    const char *model_dir;
    const char *tokenizer_json_path;
    const char *prompt_argument;
    const char *prompt_file;
    const char *tokens_out_path;
    const char *system_prompt_argument;
    const char *system_prompt_file;
    const char *reasoning_effort;
    char *owned_tokenizer_json_path;
    char *prompt_text;
    char *system_prompt_text;
    char *chat_text;
    uint32_t prompt_bytes;
    uint32_t system_prompt_bytes;
    uint32_t chat_text_bytes;
    uint32_t token_capacity;
    uint32_t chat_mode;
    uint32_t chat_flags;
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    SparkTokenizerEncoding encoding;
    FILE *tokens_out;
    SparkStatus status;
    int32_t arg_index;
    int32_t parse_status;

    model_dir = 0;
    tokenizer_json_path = 0;
    prompt_argument = 0;
    prompt_file = 0;
    tokens_out_path = 0;
    system_prompt_argument = 0;
    system_prompt_file = 0;
    reasoning_effort = "Max";
    owned_tokenizer_json_path = 0;
    prompt_text = 0;
    system_prompt_text = 0;
    chat_text = 0;
    prompt_bytes = 0u;
    system_prompt_bytes = 0u;
    chat_text_bytes = 0u;
    token_capacity = SPARK_GLM52_TOKENIZE_DEFAULT_TOKEN_CAPACITY;
    chat_mode = 0u;
    chat_flags = SPARK_GLM52_TOKENIZE_CHAT_FLAG_ADD_GENERATION_PROMPT |
        SPARK_GLM52_TOKENIZE_CHAT_FLAG_ENABLE_THINKING;

    for (arg_index = 1; arg_index < argc; ++arg_index)
    {
        if (strcmp(argv[arg_index], "--model-dir") == 0 && arg_index + 1 < argc)
        {
            model_dir = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--tokenizer-json") == 0 && arg_index + 1 < argc)
        {
            tokenizer_json_path = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--prompt") == 0 && arg_index + 1 < argc)
        {
            prompt_argument = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--prompt-file") == 0 && arg_index + 1 < argc)
        {
            prompt_file = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--tokens-out") == 0 && arg_index + 1 < argc)
        {
            tokens_out_path = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--system-prompt") == 0 && arg_index + 1 < argc)
        {
            system_prompt_argument = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--system-prompt-file") == 0 && arg_index + 1 < argc)
        {
            system_prompt_file = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--reasoning-effort") == 0 && arg_index + 1 < argc)
        {
            reasoning_effort = argv[++arg_index];
            continue;
        }
        if (strcmp(argv[arg_index], "--capacity") == 0 && arg_index + 1 < argc)
        {
            parse_status = SparkGlm52TokenizeParseU32(argv[++arg_index], &token_capacity);
            if (parse_status != 0 || token_capacity == 0u)
            {
                return SparkGlm52TokenizeUsage(argv[0]);
            }
            continue;
        }
        if (strcmp(argv[arg_index], "--chat") == 0)
        {
            chat_mode = 1u;
            continue;
        }
        if (strcmp(argv[arg_index], "--no-add-generation-prompt") == 0)
        {
            chat_flags &= ~SPARK_GLM52_TOKENIZE_CHAT_FLAG_ADD_GENERATION_PROMPT;
            continue;
        }
        if (strcmp(argv[arg_index], "--disable-thinking") == 0)
        {
            chat_flags &= ~SPARK_GLM52_TOKENIZE_CHAT_FLAG_ENABLE_THINKING;
            continue;
        }
        return SparkGlm52TokenizeUsage(argv[0]);
    }

    if ((model_dir == 0 && tokenizer_json_path == 0) ||
        (model_dir != 0 && tokenizer_json_path != 0) ||
        (prompt_argument == 0 && prompt_file == 0) ||
        (prompt_argument != 0 && prompt_file != 0) ||
        (system_prompt_argument != 0 && system_prompt_file != 0))
    {
        return SparkGlm52TokenizeUsage(argv[0]);
    }

    if (tokenizer_json_path == 0)
    {
        if (SparkGlm52TokenizeBuildTokenizerPath(
            model_dir,
            &owned_tokenizer_json_path) != 0)
        {
            fprintf(stderr, "failed to build tokenizer path\n");
            return 1;
        }
        tokenizer_json_path = owned_tokenizer_json_path;
    }

    if (prompt_argument != 0)
    {
        prompt_bytes = SparkGlm52TokenizeStringLengthU32(prompt_argument);
        prompt_text = (char *)malloc((size_t)prompt_bytes + 1u);
        if (prompt_text == 0)
        {
            free(owned_tokenizer_json_path);
            return 1;
        }
        memcpy(prompt_text, prompt_argument, prompt_bytes + 1u);
    }
    else if (SparkGlm52TokenizeReadFile(prompt_file, &prompt_text, &prompt_bytes) != 0)
    {
        fprintf(stderr, "failed to read prompt file\n");
        free(owned_tokenizer_json_path);
        return 1;
    }

    if (system_prompt_argument != 0)
    {
        system_prompt_bytes = SparkGlm52TokenizeStringLengthU32(system_prompt_argument);
        system_prompt_text = (char *)malloc((size_t)system_prompt_bytes + 1u);
        if (system_prompt_text == 0)
        {
            free(prompt_text);
            free(owned_tokenizer_json_path);
            return 1;
        }
        memcpy(system_prompt_text, system_prompt_argument, system_prompt_bytes + 1u);
    }
    else if (system_prompt_file != 0 &&
        SparkGlm52TokenizeReadFile(system_prompt_file, &system_prompt_text, &system_prompt_bytes) != 0)
    {
        fprintf(stderr, "failed to read system prompt file\n");
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }

    if (chat_mode != 0u)
    {
        if (SparkGlm52TokenizeBuildSimpleChatPrompt(
            prompt_text,
            prompt_bytes,
            system_prompt_text,
            system_prompt_bytes,
            reasoning_effort,
            chat_flags,
            &chat_text,
            &chat_text_bytes) != 0)
        {
            fprintf(stderr, "failed to render GLM52 chat prompt\n");
            free(system_prompt_text);
            free(prompt_text);
            free(owned_tokenizer_json_path);
            return 1;
        }
    }
    else
    {
        chat_text = prompt_text;
        chat_text_bytes = prompt_bytes;
        prompt_text = 0;
    }

    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = tokenizer_json_path;
    status = SparkTokenizerLoadHuggingFaceJson(&tokenizer, &configuration);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "failed to load tokenizer: %s\n", SparkStatusToString(status));
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }

    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = token_capacity;
    encoding.token_ids = (uint32_t *)malloc((size_t)token_capacity * sizeof(*encoding.token_ids));
    if (encoding.token_ids == 0)
    {
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }
    status = SparkTokenizerEncodeUtf8(
        &tokenizer,
        chat_text,
        chat_text_bytes,
        0u,
        &encoding);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "failed to encode prompt: %s\n", SparkStatusToString(status));
        free(encoding.token_ids);
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }
    if (encoding.overflow_token_count != 0u)
    {
        fprintf(stderr, "token output capacity exceeded: %u overflow tokens\n", encoding.overflow_token_count);
        free(encoding.token_ids);
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }

    tokens_out = stdout;
    if (tokens_out_path != 0)
    {
        tokens_out = fopen(tokens_out_path, "wb");
        if (tokens_out == 0)
        {
            fprintf(stderr, "failed to open token output\n");
            free(encoding.token_ids);
            SparkTokenizerDestroy(&tokenizer);
            free(chat_text);
            free(system_prompt_text);
            free(prompt_text);
            free(owned_tokenizer_json_path);
            return 1;
        }
    }
    if (SparkGlm52TokenizeWriteTokens(
        tokens_out,
        encoding.token_ids,
        encoding.token_count) != 0)
    {
        fprintf(stderr, "failed to write token output\n");
        if (tokens_out_path != 0)
        {
            fclose(tokens_out);
        }
        free(encoding.token_ids);
        SparkTokenizerDestroy(&tokenizer);
        free(chat_text);
        free(system_prompt_text);
        free(prompt_text);
        free(owned_tokenizer_json_path);
        return 1;
    }
    if (tokens_out_path != 0)
    {
        fclose(tokens_out);
    }
    fprintf(stderr,
        "sparkpipe_glm52_tokenize_token_count=%u\n",
        encoding.token_count);
    fprintf(stderr,
        "sparkpipe_glm52_tokenize_backend=c_bytelevel_bpe\n");

    free(encoding.token_ids);
    SparkTokenizerDestroy(&tokenizer);
    free(chat_text);
    free(system_prompt_text);
    free(prompt_text);
    free(owned_tokenizer_json_path);
    return 0;
}

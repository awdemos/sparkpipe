#include "sparkpipe/spark_tokenizer.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"

#define SPARK_TOKENIZER_EMPTY_BUCKET UINT32_MAX
#define SPARK_TOKENIZER_INITIAL_BYTE_SYMBOL_BYTES 4u

static uint32_t SparkTokenizerNextPowerOfTwo(
    uint32_t value)
{
    uint32_t power_of_two;

    power_of_two = 1u;
    while (power_of_two < value && power_of_two <= UINT32_MAX / 2u)
    {
        power_of_two <<= 1u;
    }
    return power_of_two;
}

static uint32_t SparkTokenizerHashBytes(
    const char *text,
    uint32_t text_bytes)
{
    uint32_t hash_value;
    uint32_t byte_index;

    hash_value = 2166136261u;
    for (byte_index = 0u; byte_index < text_bytes; ++byte_index)
    {
        hash_value ^= (uint8_t)text[byte_index];
        hash_value *= 16777619u;
    }
    return hash_value;
}

static char *SparkTokenizerDuplicateBytes(
    const char *text,
    uint32_t text_bytes)
{
    char *copy;

    if (text == 0 && text_bytes != 0u)
    {
        return 0;
    }
    copy = (char *)malloc((size_t)text_bytes + 1u);
    if (copy == 0)
    {
        return 0;
    }
    if (text_bytes != 0u)
    {
        memcpy(copy, text, text_bytes);
    }
    copy[text_bytes] = '\0';
    return copy;
}

static void SparkTokenizerInitializeBuckets(
    uint32_t *buckets,
    uint32_t bucket_count)
{
    uint32_t bucket_index;

    if (buckets == 0)
    {
        return;
    }
    for (bucket_index = 0u; bucket_index < bucket_count; ++bucket_index)
    {
        buckets[bucket_index] = SPARK_TOKENIZER_EMPTY_BUCKET;
    }
}

static SparkStatus SparkTokenizerAllocateStringTable(
    SparkTokenizerStringEntry **entries_out,
    uint32_t **buckets_out,
    uint32_t *bucket_count_out,
    uint32_t entry_count)
{
    SparkTokenizerStringEntry *entries;
    uint32_t *buckets;
    uint32_t bucket_count;

    if (entries_out == 0 || buckets_out == 0 || bucket_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *entries_out = 0;
    *buckets_out = 0;
    *bucket_count_out = 0u;
    if (entry_count == 0u)
    {
        return SPARK_STATUS_OK;
    }

    bucket_count = SparkTokenizerNextPowerOfTwo(entry_count * 2u + 1u);
    entries = (SparkTokenizerStringEntry *)calloc(entry_count, sizeof(*entries));
    buckets = (uint32_t *)malloc((uint64_t)bucket_count * sizeof(*buckets));
    if (entries == 0 || buckets == 0)
    {
        free(entries);
        free(buckets);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkTokenizerInitializeBuckets(buckets, bucket_count);

    *entries_out = entries;
    *buckets_out = buckets;
    *bucket_count_out = bucket_count;
    return SPARK_STATUS_OK;
}

static SparkTokenizerStringEntry *SparkTokenizerFindStringEntryInTable(
    SparkTokenizerStringEntry *entries,
    uint32_t *buckets,
    uint32_t bucket_count,
    const char *text,
    uint32_t text_bytes)
{
    uint32_t entry_index;

    if (entries == 0 || buckets == 0 || bucket_count == 0u || text == 0)
    {
        return 0;
    }
    entry_index = buckets[SparkTokenizerHashBytes(text, text_bytes) & (bucket_count - 1u)];
    while (entry_index != SPARK_TOKENIZER_EMPTY_BUCKET)
    {
        SparkTokenizerStringEntry *entry;

        entry = &entries[entry_index];
        if (entry->text_bytes == text_bytes && memcmp(entry->text, text, text_bytes) == 0)
        {
            return entry;
        }
        entry_index = entry->next_index;
    }
    return 0;
}

static SparkStatus SparkTokenizerInsertStringEntry(
    SparkTokenizerStringEntry *entries,
    uint32_t *buckets,
    uint32_t bucket_count,
    uint32_t entry_index,
    const char *text,
    uint32_t text_bytes,
    uint32_t value)
{
    uint32_t bucket_index;
    SparkTokenizerStringEntry *entry;

    if (entries == 0 || buckets == 0 || bucket_count == 0u || text == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkTokenizerFindStringEntryInTable(
            entries,
            buckets,
            bucket_count,
            text,
            text_bytes) != 0)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    entry = &entries[entry_index];
    entry->text = SparkTokenizerDuplicateBytes(text, text_bytes);
    if (entry->text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    entry->text_bytes = text_bytes;
    entry->value = value;
    bucket_index = SparkTokenizerHashBytes(text, text_bytes) & (bucket_count - 1u);
    entry->next_index = buckets[bucket_index];
    buckets[bucket_index] = entry_index;
    return SPARK_STATUS_OK;
}

static int32_t SparkTokenizerJsonFindNextDirectChild(
    const SparkJsonDocument *document,
    int32_t parent_token_index,
    int32_t previous_token_index)
{
    int32_t token_index;

    if (document == 0 || parent_token_index < 0)
    {
        return -1;
    }
    token_index = previous_token_index + 1;
    while (token_index >= 0 && (uint32_t)token_index < document->token_count)
    {
        if (document->tokens[token_index].parent == parent_token_index)
        {
            return token_index;
        }
        token_index += 1;
    }
    return -1;
}

static uint32_t SparkTokenizerJsonGetObjectMemberCount(
    const SparkJsonDocument *document,
    int32_t object_token_index)
{
    if (!SparkJsonTokenIsType(document, object_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        return 0u;
    }
    return document->tokens[object_token_index].child_count / 2u;
}

static SparkStatus SparkTokenizerParseVocabulary(
    SparkTokenizer *tokenizer,
    const SparkJsonDocument *document,
    int32_t vocabulary_object_token_index)
{
    int32_t key_token_index;
    uint32_t entry_index;
    SparkStatus status;

    if (tokenizer == 0 ||
        !SparkJsonTokenIsType(document, vocabulary_object_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    tokenizer->vocabulary_count = SparkTokenizerJsonGetObjectMemberCount(
        document,
        vocabulary_object_token_index);
    status = SparkTokenizerAllocateStringTable(
        &tokenizer->vocabulary_entries,
        &tokenizer->vocabulary_buckets,
        &tokenizer->vocabulary_bucket_count,
        tokenizer->vocabulary_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    entry_index = 0u;
    key_token_index = SparkTokenizerJsonFindNextDirectChild(
        document,
        vocabulary_object_token_index,
        vocabulary_object_token_index);
    while (key_token_index >= 0)
    {
        int32_t value_token_index;
        char *token_text;
        uint32_t token_id;

        value_token_index = SparkTokenizerJsonFindNextDirectChild(
            document,
            vocabulary_object_token_index,
            key_token_index);
        if (value_token_index < 0 ||
            SparkJsonCopyString(document, key_token_index, &token_text) !=
                SPARK_STATUS_OK ||
            SparkJsonGetUInt32(document, value_token_index, &token_id) !=
                SPARK_STATUS_OK)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        status = SparkTokenizerInsertStringEntry(
            tokenizer->vocabulary_entries,
            tokenizer->vocabulary_buckets,
            tokenizer->vocabulary_bucket_count,
            entry_index,
            token_text,
            (uint32_t)strlen(token_text),
            token_id);
        if (status != SPARK_STATUS_OK)
        {
            free(token_text);
            return status;
        }
        if (token_id > tokenizer->maximum_token_id)
        {
            tokenizer->maximum_token_id = token_id;
        }
        free(token_text);
        entry_index += 1u;
        key_token_index = SparkTokenizerJsonFindNextDirectChild(
            document,
            vocabulary_object_token_index,
            value_token_index);
    }
    return entry_index == tokenizer->vocabulary_count ? SPARK_STATUS_OK : SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkTokenizerBuildReverseVocabulary(
    SparkTokenizer *tokenizer)
{
    uint32_t entry_index;

    if (tokenizer == 0 || tokenizer->vocabulary_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    tokenizer->token_text_by_id = (char **)calloc(
        (uint64_t)tokenizer->maximum_token_id + 1u,
        sizeof(*tokenizer->token_text_by_id));
    tokenizer->token_text_bytes_by_id = (uint32_t *)calloc(
        (uint64_t)tokenizer->maximum_token_id + 1u,
        sizeof(*tokenizer->token_text_bytes_by_id));
    if (tokenizer->token_text_by_id == 0 || tokenizer->token_text_bytes_by_id == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    for (entry_index = 0u; entry_index < tokenizer->vocabulary_count; ++entry_index)
    {
        SparkTokenizerStringEntry *entry;

        entry = &tokenizer->vocabulary_entries[entry_index];
        if (entry->value <= tokenizer->maximum_token_id)
        {
            tokenizer->token_text_by_id[entry->value] = entry->text;
            tokenizer->token_text_bytes_by_id[entry->value] = entry->text_bytes;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerMakeMergeKey(
    const char *left_text,
    uint32_t left_text_bytes,
    const char *right_text,
    uint32_t right_text_bytes,
    char **merge_key_out,
    uint32_t *merge_key_bytes_out)
{
    char *merge_key;
    uint64_t merge_key_bytes;

    if (merge_key_out == 0 || merge_key_bytes_out == 0 ||
        left_text == 0 || right_text == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *merge_key_out = 0;
    *merge_key_bytes_out = 0u;
    merge_key_bytes = (uint64_t)left_text_bytes + 1u + right_text_bytes;
    if (merge_key_bytes > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    merge_key = (char *)malloc((size_t)merge_key_bytes + 1u);
    if (merge_key == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(merge_key, left_text, left_text_bytes);
    merge_key[left_text_bytes] = ' ';
    memcpy(merge_key + left_text_bytes + 1u, right_text, right_text_bytes);
    merge_key[merge_key_bytes] = '\0';
    *merge_key_out = merge_key;
    *merge_key_bytes_out = (uint32_t)merge_key_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerParseMergeString(
    const char *merge_text,
    uint32_t merge_text_bytes,
    char **merge_key_out,
    uint32_t *merge_key_bytes_out)
{
    uint32_t byte_index;

    if (merge_text == 0 || merge_key_out == 0 || merge_key_bytes_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (byte_index = 0u; byte_index < merge_text_bytes; ++byte_index)
    {
        if (merge_text[byte_index] == ' ')
        {
            return SparkTokenizerMakeMergeKey(
                merge_text,
                byte_index,
                merge_text + byte_index + 1u,
                merge_text_bytes - byte_index - 1u,
                merge_key_out,
                merge_key_bytes_out);
        }
    }
    return SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkTokenizerParseMergeArray(
    const SparkJsonDocument *document,
    int32_t merge_array_token_index,
    char **merge_key_out,
    uint32_t *merge_key_bytes_out)
{
    int32_t left_token_index;
    int32_t right_token_index;
    char *left_text;
    char *right_text;
    SparkStatus status;

    if (SparkJsonGetArrayElementCount(document, merge_array_token_index) != 2u)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    left_token_index = SparkJsonGetArrayElement(document, merge_array_token_index, 0u);
    right_token_index = SparkJsonGetArrayElement(document, merge_array_token_index, 1u);
    if (SparkJsonCopyString(document, left_token_index, &left_text) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    if (SparkJsonCopyString(document, right_token_index, &right_text) != SPARK_STATUS_OK)
    {
        free(left_text);
        return SPARK_STATUS_PARSE_ERROR;
    }
    status = SparkTokenizerMakeMergeKey(
        left_text,
        (uint32_t)strlen(left_text),
        right_text,
        (uint32_t)strlen(right_text),
        merge_key_out,
        merge_key_bytes_out);
    free(left_text);
    free(right_text);
    return status;
}

static SparkStatus SparkTokenizerParseMerges(
    SparkTokenizer *tokenizer,
    const SparkJsonDocument *document,
    int32_t merges_array_token_index)
{
    uint32_t merge_index;
    SparkStatus status;

    if (tokenizer == 0 ||
        !SparkJsonTokenIsType(document, merges_array_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    tokenizer->merge_count = SparkJsonGetArrayElementCount(document, merges_array_token_index);
    status = SparkTokenizerAllocateStringTable(
        &tokenizer->merge_entries,
        &tokenizer->merge_buckets,
        &tokenizer->merge_bucket_count,
        tokenizer->merge_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    for (merge_index = 0u; merge_index < tokenizer->merge_count; ++merge_index)
    {
        int32_t merge_token_index;
        char *merge_text;
        char *merge_key;
        uint32_t merge_key_bytes;

        merge_key = 0;
        merge_token_index = SparkJsonGetArrayElement(
            document,
            merges_array_token_index,
            merge_index);
        if (SparkJsonTokenIsType(document, merge_token_index, SPARK_JSON_TOKEN_STRING))
        {
            if (SparkJsonCopyString(document, merge_token_index, &merge_text) !=
                SPARK_STATUS_OK)
            {
                return SPARK_STATUS_PARSE_ERROR;
            }
            status = SparkTokenizerParseMergeString(
                merge_text,
                (uint32_t)strlen(merge_text),
                &merge_key,
                &merge_key_bytes);
            free(merge_text);
        }
        else
        {
            status = SparkTokenizerParseMergeArray(
                document,
                merge_token_index,
                &merge_key,
                &merge_key_bytes);
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkTokenizerInsertStringEntry(
            tokenizer->merge_entries,
            tokenizer->merge_buckets,
            tokenizer->merge_bucket_count,
            merge_index,
            merge_key,
            merge_key_bytes,
            merge_index);
        free(merge_key);
        if (status == SPARK_STATUS_DUPLICATE)
        {
            continue;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerParseAddedTokens(
    SparkTokenizer *tokenizer,
    const SparkJsonDocument *document,
    int32_t added_tokens_array_token_index)
{
    uint32_t array_count;
    uint32_t array_index;
    uint32_t special_count;

    if (added_tokens_array_token_index < 0 ||
        !SparkJsonTokenIsType(document, added_tokens_array_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SPARK_STATUS_OK;
    }

    array_count = SparkJsonGetArrayElementCount(document, added_tokens_array_token_index);
    tokenizer->special_tokens = (SparkTokenizerSpecialToken *)calloc(
        array_count,
        sizeof(*tokenizer->special_tokens));
    if (array_count != 0u && tokenizer->special_tokens == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    special_count = 0u;
    for (array_index = 0u; array_index < array_count; ++array_index)
    {
        int32_t token_object_index;
        int32_t id_token_index;
        int32_t content_token_index;
        int32_t special_token_index;
        bool is_special;
        uint32_t token_id;
        char *content;

        token_object_index = SparkJsonGetArrayElement(
            document,
            added_tokens_array_token_index,
            array_index);
        if (!SparkJsonTokenIsType(document, token_object_index, SPARK_JSON_TOKEN_OBJECT))
        {
            continue;
        }
        id_token_index = SparkJsonFindObjectMember(document, token_object_index, "id");
        content_token_index = SparkJsonFindObjectMember(document, token_object_index, "content");
        special_token_index = SparkJsonFindObjectMember(document, token_object_index, "special");
        is_special = false;
        if (special_token_index >= 0 &&
            SparkJsonGetBoolean(document, special_token_index, &is_special) !=
                SPARK_STATUS_OK)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (!is_special || id_token_index < 0 || content_token_index < 0)
        {
            continue;
        }
        if (SparkJsonGetUInt32(document, id_token_index, &token_id) != SPARK_STATUS_OK ||
            SparkJsonCopyString(document, content_token_index, &content) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        tokenizer->special_tokens[special_count].text = content;
        tokenizer->special_tokens[special_count].text_bytes = (uint32_t)strlen(content);
        tokenizer->special_tokens[special_count].token_id = token_id;
        special_count += 1u;
    }
    tokenizer->special_token_count = special_count;
    return SPARK_STATUS_OK;
}

static int SparkTokenizerCompareSpecialTokenLengthDescending(
    const void *left,
    const void *right)
{
    const SparkTokenizerSpecialToken *left_token;
    const SparkTokenizerSpecialToken *right_token;

    left_token = (const SparkTokenizerSpecialToken *)left;
    right_token = (const SparkTokenizerSpecialToken *)right;
    if (left_token->text_bytes != right_token->text_bytes)
    {
        return left_token->text_bytes > right_token->text_bytes ? -1 : 1;
    }
    if (left_token->token_id == right_token->token_id)
    {
        return 0;
    }
    return left_token->token_id < right_token->token_id ? -1 : 1;
}

static void SparkTokenizerSortSpecialTokens(
    SparkTokenizer *tokenizer)
{
    if (tokenizer == 0 || tokenizer->special_token_count <= 1u)
    {
        return;
    }
    qsort(
        tokenizer->special_tokens,
        tokenizer->special_token_count,
        sizeof(*tokenizer->special_tokens),
        SparkTokenizerCompareSpecialTokenLengthDescending);
}

static uint32_t SparkTokenizerJsonSearchBooleanMemberRecursive(
    const SparkJsonDocument *document,
    int32_t token_index,
    const char *member_name,
    bool *value_out)
{
    if (SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        int32_t member_token_index;
        int32_t child_token_index;

        member_token_index = SparkJsonFindObjectMember(document, token_index, member_name);
        if (member_token_index >= 0 &&
            SparkJsonGetBoolean(document, member_token_index, value_out) ==
                SPARK_STATUS_OK)
        {
            return 1u;
        }
        child_token_index = SparkTokenizerJsonFindNextDirectChild(
            document,
            token_index,
            token_index);
        while (child_token_index >= 0)
        {
            int32_t value_token_index;

            value_token_index = SparkTokenizerJsonFindNextDirectChild(
                document,
                token_index,
                child_token_index);
            if (value_token_index < 0)
            {
                break;
            }
            if (SparkTokenizerJsonSearchBooleanMemberRecursive(
                    document,
                    value_token_index,
                    member_name,
                    value_out))
            {
                return 1u;
            }
            child_token_index = SparkTokenizerJsonFindNextDirectChild(
                document,
                token_index,
                value_token_index);
        }
    }
    else if (SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        uint32_t element_count;
        uint32_t element_index;

        element_count = SparkJsonGetArrayElementCount(document, token_index);
        for (element_index = 0u; element_index < element_count; ++element_index)
        {
            if (SparkTokenizerJsonSearchBooleanMemberRecursive(
                    document,
                    SparkJsonGetArrayElement(document, token_index, element_index),
                    member_name,
                    value_out))
            {
                return 1u;
            }
        }
    }
    return 0u;
}

void SparkTokenizerReset(
    SparkTokenizer *tokenizer)
{
    if (tokenizer == 0)
    {
        return;
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
    tokenizer->abi_version = SPARK_TOKENIZER_ABI_VERSION;
    tokenizer->descriptor_bytes = SPARK_TOKENIZER_DESCRIPTOR_BYTES;
    tokenizer->model_kind = SPARK_TOKENIZER_BPE_MODEL_KIND_BYTE_LEVEL;
}

void SparkTokenizerDestroy(
    SparkTokenizer *tokenizer)
{
    uint32_t entry_index;

    if (tokenizer == 0)
    {
        return;
    }
    for (entry_index = 0u; entry_index < tokenizer->vocabulary_count; ++entry_index)
    {
        free(tokenizer->vocabulary_entries[entry_index].text);
    }
    for (entry_index = 0u; entry_index < tokenizer->merge_count; ++entry_index)
    {
        free(tokenizer->merge_entries[entry_index].text);
    }
    for (entry_index = 0u; entry_index < tokenizer->special_token_count; ++entry_index)
    {
        free(tokenizer->special_tokens[entry_index].text);
    }
    free(tokenizer->vocabulary_entries);
    free(tokenizer->vocabulary_buckets);
    free(tokenizer->merge_entries);
    free(tokenizer->merge_buckets);
    free(tokenizer->special_tokens);
    free(tokenizer->token_text_by_id);
    free(tokenizer->token_text_bytes_by_id);
    SparkTokenizerReset(tokenizer);
}

void SparkTokenizerEncodingReset(
    SparkTokenizerEncoding *encoding)
{
    uint32_t token_capacity;
    uint32_t *token_ids;

    if (encoding == 0)
    {
        return;
    }
    token_capacity = encoding->token_capacity;
    token_ids = encoding->token_ids;
    memset(encoding, 0, sizeof(*encoding));
    encoding->abi_version = SPARK_TOKENIZER_ABI_VERSION;
    encoding->descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
    encoding->token_capacity = token_capacity;
    encoding->token_ids = token_ids;
}

SparkStatus SparkTokenizerFindTokenId(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t *token_id_out)
{
    SparkTokenizerStringEntry *entry;

    if (tokenizer == 0 || text == 0 || token_id_out == 0 ||
        tokenizer->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        tokenizer->descriptor_bytes != SPARK_TOKENIZER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *token_id_out = 0u;
    entry = SparkTokenizerFindStringEntryInTable(
        tokenizer->vocabulary_entries,
        tokenizer->vocabulary_buckets,
        tokenizer->vocabulary_bucket_count,
        text,
        text_bytes);
    if (entry == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    *token_id_out = entry->value;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerLoadHuggingFaceJson(
    SparkTokenizer *tokenizer,
    const SparkTokenizerHuggingFaceJsonConfiguration *configuration)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t model_token_index;
    int32_t vocabulary_token_index;
    int32_t merges_token_index;
    int32_t added_tokens_token_index;
    int32_t unknown_token_index;
    int32_t byte_fallback_token_index;
    int32_t pre_tokenizer_token_index;
    bool add_prefix_space;
    bool byte_fallback;
    SparkStatus status;

    if (tokenizer == 0 ||
        configuration == 0 ||
        configuration->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->tokenizer_json_path == 0 ||
        configuration->reserved0 != 0u ||
        configuration->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkTokenizerReset(tokenizer);
    SparkJsonDocumentReset(&document);
    status = SparkJsonLoadFile(configuration->tokenizer_json_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    root_token_index = SparkJsonGetRootToken(&document);
    model_token_index = SparkJsonFindObjectMember(&document, root_token_index, "model");
    if (model_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        SparkTokenizerDestroy(tokenizer);
        return SPARK_STATUS_PARSE_ERROR;
    }
    vocabulary_token_index = SparkJsonFindObjectMember(&document, model_token_index, "vocab");
    merges_token_index = SparkJsonFindObjectMember(&document, model_token_index, "merges");
    if (vocabulary_token_index < 0 || merges_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        SparkTokenizerDestroy(tokenizer);
        return SPARK_STATUS_PARSE_ERROR;
    }

    status = SparkTokenizerParseVocabulary(tokenizer, &document, vocabulary_token_index);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerBuildReverseVocabulary(tokenizer);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerParseMerges(tokenizer, &document, merges_token_index);
    }
    added_tokens_token_index = SparkJsonFindObjectMember(&document, root_token_index, "added_tokens");
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerParseAddedTokens(tokenizer, &document, added_tokens_token_index);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkJsonDocumentDestroy(&document);
        SparkTokenizerDestroy(tokenizer);
        return status;
    }

    tokenizer->has_unk_token = 0u;
    tokenizer->unk_token_id = 0u;
    unknown_token_index = SparkJsonFindObjectMember(&document, model_token_index, "unk_token");
    if (unknown_token_index >= 0)
    {
        char *unknown_token;
        uint32_t unknown_token_id;

        if (SparkJsonCopyString(&document, unknown_token_index, &unknown_token) ==
            SPARK_STATUS_OK)
        {
            if (SparkTokenizerFindTokenId(
                    tokenizer,
                    unknown_token,
                    (uint32_t)strlen(unknown_token),
                    &unknown_token_id) == SPARK_STATUS_OK)
            {
                tokenizer->has_unk_token = 1u;
                tokenizer->unk_token_id = unknown_token_id;
            }
            free(unknown_token);
        }
    }

    byte_fallback = false;
    byte_fallback_token_index = SparkJsonFindObjectMember(&document, model_token_index, "byte_fallback");
    if (byte_fallback_token_index >= 0)
    {
        (void)SparkJsonGetBoolean(&document, byte_fallback_token_index, &byte_fallback);
    }
    tokenizer->byte_fallback = byte_fallback ? 1u : 0u;

    add_prefix_space = false;
    pre_tokenizer_token_index = SparkJsonFindObjectMember(&document, root_token_index, "pre_tokenizer");
    if (pre_tokenizer_token_index >= 0)
    {
        (void)SparkTokenizerJsonSearchBooleanMemberRecursive(
            &document,
            pre_tokenizer_token_index,
            "add_prefix_space",
            &add_prefix_space);
    }
    tokenizer->add_prefix_space = add_prefix_space ? 1u : 0u;
    SparkTokenizerSortSpecialTokens(tokenizer);
    SparkJsonDocumentDestroy(&document);
    return SPARK_STATUS_OK;
}

typedef struct SparkTokenizerEncodeSymbol
{
    char *text;
    uint32_t text_bytes;
    uint32_t token_id;
} SparkTokenizerEncodeSymbol;

static uint32_t SparkTokenizerByteToUnicodeCodePoint(
    uint32_t byte_value)
{
    uint32_t candidate;
    uint32_t next_code_point;

    if ((byte_value >= 33u && byte_value <= 126u) ||
        (byte_value >= 161u && byte_value <= 172u) ||
        (byte_value >= 174u && byte_value <= 255u))
    {
        return byte_value;
    }

    next_code_point = 256u;
    for (candidate = 0u; candidate <= byte_value; ++candidate)
    {
        if (!((candidate >= 33u && candidate <= 126u) ||
              (candidate >= 161u && candidate <= 172u) ||
              (candidate >= 174u && candidate <= 255u)))
        {
            if (candidate == byte_value)
            {
                return next_code_point;
            }
            next_code_point += 1u;
        }
    }
    return byte_value;
}

static uint32_t SparkTokenizerAppendUtf8CodePoint(
    char *destination,
    uint32_t code_point)
{
    if (code_point <= 0x7fu)
    {
        destination[0u] = (char)code_point;
        return 1u;
    }
    if (code_point <= 0x7ffu)
    {
        destination[0u] = (char)(0xc0u | (code_point >> 6u));
        destination[1u] = (char)(0x80u | (code_point & 0x3fu));
        return 2u;
    }
    destination[0u] = (char)(0xe0u | (code_point >> 12u));
    destination[1u] = (char)(0x80u | ((code_point >> 6u) & 0x3fu));
    destination[2u] = (char)(0x80u | (code_point & 0x3fu));
    return 3u;
}

static void SparkTokenizerFreeSymbols(
    SparkTokenizerEncodeSymbol *symbols,
    uint32_t symbol_count)
{
    uint32_t symbol_index;

    if (symbols == 0)
    {
        return;
    }
    for (symbol_index = 0u; symbol_index < symbol_count; ++symbol_index)
    {
        free(symbols[symbol_index].text);
    }
    free(symbols);
}

static SparkStatus SparkTokenizerAppendTokenToEncoding(
    SparkTokenizerEncoding *encoding,
    uint32_t token_id)
{
    if (encoding == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (encoding->token_count < encoding->token_capacity)
    {
        encoding->token_ids[encoding->token_count] = token_id;
        encoding->token_count += 1u;
    }
    else
    {
        encoding->overflow_token_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkTokenizerFindSpecialTokenAt(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t remaining_text_bytes,
    uint32_t *token_id_out,
    uint32_t *matched_text_bytes_out)
{
    uint32_t special_token_index;

    if (tokenizer == 0 || text == 0 || token_id_out == 0 ||
        matched_text_bytes_out == 0)
    {
        return 0u;
    }
    for (special_token_index = 0u;
         special_token_index < tokenizer->special_token_count;
         ++special_token_index)
    {
        const SparkTokenizerSpecialToken *special_token;

        special_token = &tokenizer->special_tokens[special_token_index];
        if (special_token->text_bytes <= remaining_text_bytes &&
            memcmp(text, special_token->text, special_token->text_bytes) == 0)
        {
            *token_id_out = special_token->token_id;
            *matched_text_bytes_out = special_token->text_bytes;
            return 1u;
        }
    }
    return 0u;
}

static SparkStatus SparkTokenizerCreateByteSymbol(
    const SparkTokenizer *tokenizer,
    uint8_t byte_value,
    SparkTokenizerEncodeSymbol *symbol)
{
    char encoded_text[SPARK_TOKENIZER_INITIAL_BYTE_SYMBOL_BYTES];
    uint32_t encoded_text_bytes;
    uint32_t token_id;
    SparkStatus status;

    if (symbol == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(symbol, 0, sizeof(*symbol));
    encoded_text_bytes = SparkTokenizerAppendUtf8CodePoint(
        encoded_text,
        SparkTokenizerByteToUnicodeCodePoint(byte_value));
    symbol->text = SparkTokenizerDuplicateBytes(encoded_text, encoded_text_bytes);
    if (symbol->text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    symbol->text_bytes = encoded_text_bytes;
    status = SparkTokenizerFindTokenId(
        tokenizer,
        symbol->text,
        symbol->text_bytes,
        &token_id);
    if (status != SPARK_STATUS_OK)
    {
        if (tokenizer->has_unk_token == 0u)
        {
            free(symbol->text);
            memset(symbol, 0, sizeof(*symbol));
            return status;
        }
        token_id = tokenizer->unk_token_id;
    }
    symbol->token_id = token_id;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerConcatenateSymbols(
    const SparkTokenizerEncodeSymbol *left,
    const SparkTokenizerEncodeSymbol *right,
    SparkTokenizerEncodeSymbol *merged)
{
    uint64_t merged_text_bytes;

    if (left == 0 || right == 0 || merged == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(merged, 0, sizeof(*merged));
    merged_text_bytes = (uint64_t)left->text_bytes + right->text_bytes;
    if (merged_text_bytes > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    merged->text = (char *)malloc((size_t)merged_text_bytes + 1u);
    if (merged->text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(merged->text, left->text, left->text_bytes);
    memcpy(merged->text + left->text_bytes, right->text, right->text_bytes);
    merged->text[merged_text_bytes] = '\0';
    merged->text_bytes = (uint32_t)merged_text_bytes;
    return SPARK_STATUS_OK;
}

static uint32_t SparkTokenizerLookupMergeRank(
    const SparkTokenizer *tokenizer,
    const SparkTokenizerEncodeSymbol *left,
    const SparkTokenizerEncodeSymbol *right,
    uint32_t *rank_out)
{
    char *merge_key;
    uint32_t merge_key_bytes;
    SparkTokenizerStringEntry *entry;
    SparkStatus status;

    if (tokenizer == 0 || left == 0 || right == 0 || rank_out == 0)
    {
        return 0u;
    }
    status = SparkTokenizerMakeMergeKey(
        left->text,
        left->text_bytes,
        right->text,
        right->text_bytes,
        &merge_key,
        &merge_key_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    entry = SparkTokenizerFindStringEntryInTable(
        tokenizer->merge_entries,
        tokenizer->merge_buckets,
        tokenizer->merge_bucket_count,
        merge_key,
        merge_key_bytes);
    free(merge_key);
    if (entry == 0)
    {
        return 0u;
    }
    *rank_out = entry->value;
    return 1u;
}

static SparkStatus SparkTokenizerApplyMergesToSymbols(
    const SparkTokenizer *tokenizer,
    SparkTokenizerEncodeSymbol *symbols,
    uint32_t *symbol_count)
{
    if (tokenizer == 0 || symbols == 0 || symbol_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    while (*symbol_count > 1u)
    {
        uint32_t symbol_index;
        uint32_t best_symbol_index;
        uint32_t best_rank;
        SparkTokenizerEncodeSymbol merged_symbol;
        SparkStatus status;

        best_symbol_index = UINT32_MAX;
        best_rank = UINT32_MAX;
        for (symbol_index = 0u; symbol_index + 1u < *symbol_count; ++symbol_index)
        {
            uint32_t rank;

            if (SparkTokenizerLookupMergeRank(
                    tokenizer,
                    &symbols[symbol_index],
                    &symbols[symbol_index + 1u],
                    &rank) &&
                rank < best_rank)
            {
                best_rank = rank;
                best_symbol_index = symbol_index;
            }
        }
        if (best_symbol_index == UINT32_MAX)
        {
            break;
        }

        status = SparkTokenizerConcatenateSymbols(
            &symbols[best_symbol_index],
            &symbols[best_symbol_index + 1u],
            &merged_symbol);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkTokenizerFindTokenId(
            tokenizer,
            merged_symbol.text,
            merged_symbol.text_bytes,
            &merged_symbol.token_id);
        if (status != SPARK_STATUS_OK)
        {
            free(merged_symbol.text);
            return status;
        }

        free(symbols[best_symbol_index].text);
        free(symbols[best_symbol_index + 1u].text);
        symbols[best_symbol_index] = merged_symbol;
        if (best_symbol_index + 2u < *symbol_count)
        {
            memmove(
                &symbols[best_symbol_index + 1u],
                &symbols[best_symbol_index + 2u],
                (size_t)(*symbol_count - best_symbol_index - 2u) * sizeof(*symbols));
        }
        *symbol_count -= 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerEncodeRegularSegment(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    SparkTokenizerEncoding *encoding)
{
    SparkTokenizerEncodeSymbol *symbols;
    uint32_t symbol_count;
    uint32_t symbol_index;
    SparkStatus status;

    if (text_bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    symbols = (SparkTokenizerEncodeSymbol *)calloc(text_bytes, sizeof(*symbols));
    if (symbols == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    symbol_count = 0u;
    for (symbol_index = 0u; symbol_index < text_bytes; ++symbol_index)
    {
        status = SparkTokenizerCreateByteSymbol(
            tokenizer,
            (uint8_t)text[symbol_index],
            &symbols[symbol_count]);
        if (status != SPARK_STATUS_OK)
        {
            SparkTokenizerFreeSymbols(symbols, symbol_count);
            encoding->invalid_segment_count += 1u;
            return status;
        }
        symbol_count += 1u;
    }

    status = SparkTokenizerApplyMergesToSymbols(tokenizer, symbols, &symbol_count);
    if (status != SPARK_STATUS_OK)
    {
        SparkTokenizerFreeSymbols(symbols, symbol_count);
        encoding->invalid_segment_count += 1u;
        return status;
    }
    for (symbol_index = 0u; symbol_index < symbol_count; ++symbol_index)
    {
        status = SparkTokenizerAppendTokenToEncoding(encoding, symbols[symbol_index].token_id);
        if (status != SPARK_STATUS_OK)
        {
            SparkTokenizerFreeSymbols(symbols, symbol_count);
            return status;
        }
    }
    SparkTokenizerFreeSymbols(symbols, symbol_count);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerEncodeUtf8(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerEncoding *encoding)
{
    uint32_t position;
    uint32_t segment_start;
    uint32_t add_prefix_space;
    SparkStatus status;

    if (tokenizer == 0 ||
        tokenizer->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        tokenizer->descriptor_bytes != SPARK_TOKENIZER_DESCRIPTOR_BYTES ||
        (text == 0 && text_bytes != 0u) ||
        (encode_flags & ~SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS) != 0u ||
        encoding == 0 ||
        encoding->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        encoding->descriptor_bytes != SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES ||
        (encoding->token_ids == 0 && encoding->token_capacity != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    encoding->token_count = 0u;
    encoding->overflow_token_count = 0u;
    encoding->invalid_segment_count = 0u;
    add_prefix_space = ((encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE) != 0u ||
        tokenizer->add_prefix_space != 0u) &&
        (text_bytes == 0u || text[0u] != ' ');
    if (add_prefix_space != 0u)
    {
        status = SparkTokenizerEncodeRegularSegment(tokenizer, " ", 1u, encoding);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    position = 0u;
    segment_start = 0u;
    while (position < text_bytes)
    {
        uint32_t special_token_id;
        uint32_t matched_text_bytes;

        if ((encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH) == 0u &&
            SparkTokenizerFindSpecialTokenAt(
                tokenizer,
                text + position,
                text_bytes - position,
                &special_token_id,
                &matched_text_bytes))
        {
            status = SparkTokenizerEncodeRegularSegment(
                tokenizer,
                text + segment_start,
                position - segment_start,
                encoding);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            status = SparkTokenizerAppendTokenToEncoding(encoding, special_token_id);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            position += matched_text_bytes;
            segment_start = position;
        }
        else
        {
            position += 1u;
        }
    }
    status = SparkTokenizerEncodeRegularSegment(
        tokenizer,
        text + segment_start,
        text_bytes - segment_start,
        encoding);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return encoding->overflow_token_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
}

SparkStatus SparkTokenizerEncodeBatchUtf8(
    const SparkTokenizer *tokenizer,
    const char *const *texts,
    const uint32_t *text_bytes,
    uint32_t text_count,
    uint32_t encode_flags,
    uint32_t *token_ids,
    uint32_t token_stride,
    uint32_t *token_counts,
    uint32_t *overflow_token_counts)
{
    uint32_t text_index;
    SparkStatus final_status;

    if (texts == 0 || text_bytes == 0 ||
        (token_ids == 0 && token_stride != 0u) ||
        token_counts == 0 || overflow_token_counts == 0 ||
        (text_count != 0u && token_stride == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    final_status = SPARK_STATUS_OK;
    for (text_index = 0u; text_index < text_count; ++text_index)
    {
        SparkTokenizerEncoding encoding;
        SparkStatus status;

        memset(&encoding, 0, sizeof(encoding));
        encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        encoding.descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
        encoding.token_capacity = token_stride;
        encoding.token_ids = &token_ids[(uint64_t)text_index * token_stride];
        status = SparkTokenizerEncodeUtf8(
            tokenizer,
            texts[text_index],
            text_bytes[text_index],
            encode_flags,
            &encoding);
        token_counts[text_index] = encoding.token_count;
        overflow_token_counts[text_index] = encoding.overflow_token_count;
        if (status != SPARK_STATUS_OK && final_status == SPARK_STATUS_OK)
        {
            final_status = status;
        }
    }
    return final_status;
}

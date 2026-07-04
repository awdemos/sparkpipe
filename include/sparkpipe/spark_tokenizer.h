#ifndef SPARKPIPE_SPARK_TOKENIZER_H
#define SPARKPIPE_SPARK_TOKENIZER_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TOKENIZER_ABI_VERSION 1u
#define SPARK_TOKENIZER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizer))
#define SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerHuggingFaceJsonConfiguration))
#define SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerEncoding))
#define SPARK_TOKENIZER_BPE_MODEL_KIND_BYTE_LEVEL 1u
#define SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES 256u

#define SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH 0x00000001u
#define SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE 0x00000002u
#define SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS \
    (SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH | \
     SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE)

typedef struct SparkTokenizerStringEntry
{
    char *text;
    uint32_t text_bytes;
    uint32_t value;
    uint32_t next_index;
} SparkTokenizerStringEntry;

typedef struct SparkTokenizerSpecialToken
{
    char *text;
    uint32_t text_bytes;
    uint32_t token_id;
    uint32_t reserved0;
} SparkTokenizerSpecialToken;

typedef struct SparkTokenizerHuggingFaceJsonConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *tokenizer_json_path;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerHuggingFaceJsonConfiguration;

typedef struct SparkTokenizerEncoding
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t token_count;
    uint32_t overflow_token_count;
    uint32_t invalid_segment_count;
    uint32_t token_capacity;
    uint32_t *token_ids;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerEncoding;

typedef struct SparkTokenizer
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t model_kind;
    uint32_t add_prefix_space;
    uint32_t byte_fallback;
    uint32_t has_unk_token;
    uint32_t unk_token_id;
    uint32_t maximum_token_id;
    uint32_t reserved2;
    char **token_text_by_id;
    uint32_t *token_text_bytes_by_id;
    uint32_t vocabulary_count;
    SparkTokenizerStringEntry *vocabulary_entries;
    uint32_t *vocabulary_buckets;
    uint32_t vocabulary_bucket_count;
    uint32_t merge_count;
    SparkTokenizerStringEntry *merge_entries;
    uint32_t *merge_buckets;
    uint32_t merge_bucket_count;
    uint32_t special_token_count;
    SparkTokenizerSpecialToken *special_tokens;
} SparkTokenizer;

void SparkTokenizerReset(
    SparkTokenizer *tokenizer);

void SparkTokenizerDestroy(
    SparkTokenizer *tokenizer);

void SparkTokenizerEncodingReset(
    SparkTokenizerEncoding *encoding);

SparkStatus SparkTokenizerFindTokenId(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t *token_id_out);

SparkStatus SparkTokenizerLoadHuggingFaceJson(
    SparkTokenizer *tokenizer,
    const SparkTokenizerHuggingFaceJsonConfiguration *configuration);

SparkStatus SparkTokenizerEncodeUtf8(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerEncoding *encoding);

SparkStatus SparkTokenizerEncodeBatchUtf8(
    const SparkTokenizer *tokenizer,
    const char *const *texts,
    const uint32_t *text_bytes,
    uint32_t text_count,
    uint32_t encode_flags,
    uint32_t *token_ids,
    uint32_t token_stride,
    uint32_t *token_counts,
    uint32_t *overflow_token_counts);

#ifdef __cplusplus
}
#endif

#endif

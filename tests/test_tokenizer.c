#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tokenizer.h"

#define SPARK_TEST_TOKEN_A 1u
#define SPARK_TEST_TOKEN_B 2u
#define SPARK_TEST_TOKEN_C 3u
#define SPARK_TEST_TOKEN_AB 4u
#define SPARK_TEST_TOKEN_ABC 5u
#define SPARK_TEST_TOKEN_UNKNOWN 6u
#define SPARK_TEST_TOKEN_STOP 7u
#define SPARK_TEST_TOKEN_SPACE 8u
#define SPARK_TEST_TOKEN_X 9u
#define SPARK_TEST_TOKEN_Y 10u
#define SPARK_TEST_TOKEN_Z 11u
#define SPARK_TEST_TOKEN_XY 12u
#define SPARK_TEST_TOKEN_XYZ 13u

static const char *SparkTestTokenizerJsonPath(void)
{
    return "build/test_tokenizer_hf_byte_bpe.json";
}

static void SparkTestTokenizerWriteFixtureJson(void)
{
    FILE *file;

    file = fopen(SparkTestTokenizerJsonPath(), "wb");
    assert(file != 0);
    fprintf(file,
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"unk_token\": \"<unk>\",\n"
        "    \"byte_fallback\": false,\n"
        "    \"vocab\": {\n"
        "      \"a\": %u,\n"
        "      \"b\": %u,\n"
        "      \"c\": %u,\n"
        "      \"ab\": %u,\n"
        "      \"abc\": %u,\n"
        "      \"<unk>\": %u,\n"
        "      \"<|stop|>\": %u,\n"
        "      \"\304\240\": %u,\n"
        "      \"x\": %u,\n"
        "      \"y\": %u,\n"
        "      \"z\": %u,\n"
        "      \"xy\": %u,\n"
        "      \"xyz\": %u\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"a b\",\n"
        "      \"ab c\",\n"
        "      \"x y\",\n"
        "      \"xy z\"\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"ByteLevel\",\n"
        "    \"add_prefix_space\": false\n"
        "  },\n"
        "  \"added_tokens\": [\n"
        "    {\"id\": %u, \"content\": \"<|stop|>\", \"special\": true}\n"
        "  ]\n"
        "}\n",
        SPARK_TEST_TOKEN_A,
        SPARK_TEST_TOKEN_B,
        SPARK_TEST_TOKEN_C,
        SPARK_TEST_TOKEN_AB,
        SPARK_TEST_TOKEN_ABC,
        SPARK_TEST_TOKEN_UNKNOWN,
        SPARK_TEST_TOKEN_STOP,
        SPARK_TEST_TOKEN_SPACE,
        SPARK_TEST_TOKEN_X,
        SPARK_TEST_TOKEN_Y,
        SPARK_TEST_TOKEN_Z,
        SPARK_TEST_TOKEN_XY,
        SPARK_TEST_TOKEN_XYZ,
        SPARK_TEST_TOKEN_STOP);
    assert(fclose(file) == 0);
}

static void SparkTestTokenizerLoadFixture(
    SparkTokenizer *tokenizer)
{
    SparkTokenizerHuggingFaceJsonConfiguration configuration;

    SparkTokenizerReset(tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = SparkTestTokenizerJsonPath();
    assert(SparkTokenizerLoadHuggingFaceJson(tokenizer, &configuration) ==
        SPARK_STATUS_OK);
}

static void SparkTestTokenizerEncodesByteBpeAndSpecialTokens(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerEncoding encoding;
    uint32_t token_ids[16u];

    SparkTestTokenizerWriteFixtureJson();
    SparkTestTokenizerLoadFixture(&tokenizer);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "abc",
        3u,
        0u,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 1u);
    assert(encoding.overflow_token_count == 0u);
    assert(encoding.invalid_segment_count == 0u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_ABC);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "a<|stop|>bc",
        11u,
        0u,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 4u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_A);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_STOP);
    assert(token_ids[2u] == SPARK_TEST_TOKEN_B);
    assert(token_ids[3u] == SPARK_TEST_TOKEN_C);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "abc",
        3u,
        SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 2u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_SPACE);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_ABC);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 2u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "a<|stop|>bc",
        11u,
        0u,
        &encoding) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(encoding.token_count == 2u);
    assert(encoding.overflow_token_count == 2u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_A);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_STOP);

    SparkTokenizerDestroy(&tokenizer);
}

static void SparkTestTokenizerEncodesBatch(void)
{
    SparkTokenizer tokenizer;
    const char *texts[2u];
    uint32_t text_bytes[2u];
    uint32_t token_ids[8u];
    uint32_t token_counts[2u];
    uint32_t overflow_counts[2u];

    SparkTestTokenizerLoadFixture(&tokenizer);
    memset(token_ids, 0, sizeof(token_ids));
    memset(token_counts, 0, sizeof(token_counts));
    memset(overflow_counts, 0, sizeof(overflow_counts));

    texts[0u] = "abc";
    text_bytes[0u] = 3u;
    texts[1u] = "xyz";
    text_bytes[1u] = 3u;
    assert(SparkTokenizerEncodeBatchUtf8(
        &tokenizer,
        texts,
        text_bytes,
        2u,
        0u,
        token_ids,
        4u,
        token_counts,
        overflow_counts) == SPARK_STATUS_OK);
    assert(token_counts[0u] == 1u);
    assert(token_counts[1u] == 1u);
    assert(overflow_counts[0u] == 0u);
    assert(overflow_counts[1u] == 0u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_ABC);
    assert(token_ids[4u] == SPARK_TEST_TOKEN_XYZ);

    SparkTokenizerDestroy(&tokenizer);
}

int main(void)
{
    SparkTestTokenizerEncodesByteBpeAndSpecialTokens();
    SparkTestTokenizerEncodesBatch();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_glm52_pp13_runtime.h"
#include "sparkpipe/spark_json.h"

#define SPARK_GLM52_PP13_GATE_DEFAULT_MAX_ACTIVE 1024u

typedef struct SparkGlm52Pp13GateConfig
{
    const char *config_path;
    const char *fp8_pack_root;
    const char *generated_root;
    uint32_t rank_index;
    uint32_t rank_is_set;
    uint32_t max_active_sequence_count;
    uint32_t port_base;
    uint32_t require_pack_files;
    uint32_t require_gpudirect_preflight;
    uint32_t generate_missing_runtime_files;
} SparkGlm52Pp13GateConfig;

static int SparkGlm52Pp13GateParseU32(const char *text,uint32_t *value_out)
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
        value = (value * 10u) + (uint32_t)(text[index] - '0');
        if (value > 0xffffffffull)
        {
            return -3;
        }
    }
    *value_out = (uint32_t)value;
    return 0;
}

static void SparkGlm52Pp13GateInitializeConfig(
    SparkGlm52Pp13GateConfig *configuration)
{
    memset(configuration,0,sizeof(*configuration));
    configuration->max_active_sequence_count =
        SPARK_GLM52_PP13_GATE_DEFAULT_MAX_ACTIVE;
    configuration->port_base = SPARK_GLM52_PP13_RUNTIME_DEFAULT_PORT_BASE;
    configuration->require_pack_files = 1u;
    configuration->require_gpudirect_preflight = 0u;
    configuration->generate_missing_runtime_files = 1u;
}

static SparkStatus SparkGlm52Pp13GateGetOptionalU32(
    const SparkJsonDocument *document,
    int32_t root,
    const char *name,
    uint32_t *value)
{
    int32_t token_index;

    token_index = SparkJsonFindObjectMember(document,root,name);
    if (token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    return SparkJsonGetUInt32(document,token_index,value);
}

static SparkStatus SparkGlm52Pp13GateGetOptionalStringPointer(
    const SparkJsonDocument *document,
    int32_t root,
    const char *name,
    const char **value)
{
    const SparkJsonToken *token;
    int32_t token_index;

    token_index = SparkJsonFindObjectMember(document,root,name);
    if (token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document,token_index,SPARK_JSON_TOKEN_STRING))
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    token = &document->tokens[token_index];
    ((char *)document->text)[token->end] = '\0';
    *value = document->text + token->start;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13GateLoadConfigFile(
    SparkGlm52Pp13GateConfig *configuration,
    SparkJsonDocument *document,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;
    int32_t root;
    uint32_t value;

    if (configuration->config_path == 0)
    {
        return SPARK_STATUS_OK;
    }
    SparkJsonDocumentReset(document);
    status = SparkJsonLoadFile(configuration->config_path,document);
    if (status != SPARK_STATUS_OK)
    {
        (void)snprintf(error_buffer,error_buffer_bytes,"failed to load config");
        return status;
    }
    root = SparkJsonGetRootToken(document);
    if (!SparkJsonTokenIsType(document,root,SPARK_JSON_TOKEN_OBJECT))
    {
        (void)snprintf(error_buffer,error_buffer_bytes,"config root is not an object");
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkGlm52Pp13GateGetOptionalStringPointer(
        document,root,"fp8_pack_root",&configuration->fp8_pack_root);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52Pp13GateGetOptionalStringPointer(
        document,root,"generated_root",&configuration->generated_root);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52Pp13GateGetOptionalU32(
        document,root,"max_active_sequence_count",
        &configuration->max_active_sequence_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52Pp13GateGetOptionalU32(
        document,root,"port_base",&configuration->port_base);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    value = configuration->require_pack_files;
    status = SparkGlm52Pp13GateGetOptionalU32(
        document,root,"require_pack_files",&value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    configuration->require_pack_files = value != 0u ? 1u : 0u;
    value = configuration->require_gpudirect_preflight;
    status = SparkGlm52Pp13GateGetOptionalU32(
        document,root,"require_gpudirect_preflight",&value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    configuration->require_gpudirect_preflight = value != 0u ? 1u : 0u;
    value = configuration->generate_missing_runtime_files;
    status = SparkGlm52Pp13GateGetOptionalU32(
        document,root,"generate_missing_runtime_files",&value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    configuration->generate_missing_runtime_files = value != 0u ? 1u : 0u;
    return SPARK_STATUS_OK;
}

static int SparkGlm52Pp13GateApplyArgument(
    SparkGlm52Pp13GateConfig *configuration,
    int argc,
    char **argv,
    int *index)
{
    uint32_t parsed;

    if (strcmp(argv[*index],"--config") == 0)
    {
        if (*index + 1 >= argc)
        {
            return -1;
        }
        configuration->config_path = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--rank") == 0)
    {
        if (*index + 1 >= argc ||
            SparkGlm52Pp13GateParseU32(argv[*index + 1],&parsed) < 0)
        {
            return -2;
        }
        configuration->rank_index = parsed;
        configuration->rank_is_set = 1u;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--fp8-pack-root") == 0)
    {
        if (*index + 1 >= argc)
        {
            return -3;
        }
        configuration->fp8_pack_root = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--generated-root") == 0)
    {
        if (*index + 1 >= argc)
        {
            return -4;
        }
        configuration->generated_root = argv[*index + 1];
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--max-active") == 0)
    {
        if (*index + 1 >= argc ||
            SparkGlm52Pp13GateParseU32(argv[*index + 1],&parsed) < 0)
        {
            return -5;
        }
        configuration->max_active_sequence_count = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--port-base") == 0)
    {
        if (*index + 1 >= argc ||
            SparkGlm52Pp13GateParseU32(argv[*index + 1],&parsed) < 0)
        {
            return -6;
        }
        configuration->port_base = parsed;
        *index += 1;
        return 0;
    }
    if (strcmp(argv[*index],"--no-pack-file-check") == 0)
    {
        configuration->require_pack_files = 0u;
        return 0;
    }
    if (strcmp(argv[*index],"--require-gpudirect-preflight") == 0)
    {
        configuration->require_gpudirect_preflight = 1u;
        return 0;
    }
    if (strcmp(argv[*index],"--no-generate-runtime-files") == 0)
    {
        configuration->generate_missing_runtime_files = 0u;
        return 0;
    }
    return -7;
}

static uint32_t SparkGlm52Pp13GateFileExists(const char *path)
{
    struct stat path_status;

    if (path == 0 || path[0] == '\0')
    {
        return 0u;
    }
    return stat(path,&path_status) == 0 ? 1u : 0u;
}

static SparkStatus SparkGlm52Pp13GateEnsureDirectory(const char *path)
{
    struct stat path_status;

    if (path == 0 || path[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (stat(path,&path_status) == 0)
    {
        return S_ISDIR(path_status.st_mode) ? SPARK_STATUS_OK :
            SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (mkdir(path,0775) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13GateBuildRankDirectory(
    const char *generated_root,
    uint32_t rank_index,
    char *rank_directory,
    uint32_t rank_directory_bytes)
{
    int written;

    if (generated_root == 0 || generated_root[0] == '\0' ||
        rank_directory == 0 || rank_directory_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    written = snprintf(
        rank_directory,rank_directory_bytes,"%s/rank_%02u",
        generated_root,rank_index);
    if (written < 0 || (uint32_t)written >= rank_directory_bytes)
    {
        rank_directory[0] = '\0';
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13GateWritePackList(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    const char *fp8_pack_root,
    const char *path)
{
    char pack_path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    FILE *file;
    SparkStatus status;
    uint32_t layer_index;

    if (rank_plan == 0 || fp8_pack_root == 0 || path == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52Pp13GateFileExists(path))
    {
        return SPARK_STATUS_OK;
    }
    file = fopen(path,"wb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    fprintf(file,"layer\tpath\n");
    for (layer_index = rank_plan->first_layer_index;
         layer_index < rank_plan->first_layer_index + rank_plan->layer_count;
         ++layer_index)
    {
        if (layer_index < SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
        {
            continue;
        }
        status = SparkGlm52Pp13RuntimeBuildFp8PackPath(
            fp8_pack_root,layer_index,pack_path,sizeof(pack_path));
        if (status != SPARK_STATUS_OK)
        {
            fclose(file);
            return status;
        }
        fprintf(file,"%u\t%s\n",layer_index,pack_path);
    }
    if (fclose(file) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13GateWriteRankManifest(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    const char *fp8_pack_root,
    const char *path)
{
    FILE *file;

    if (rank_plan == 0 || path == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52Pp13GateFileExists(path))
    {
        return SPARK_STATUS_OK;
    }
    file = fopen(path,"wb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    fprintf(file,"{\n");
    fprintf(file,"  \"format\": \"sparkpipe.glm52.pp13.fp8.rank_runtime.v1\",\n");
    fprintf(file,"  \"rank\": %u,\n",rank_plan->rank_index);
    fprintf(file,"  \"host\": \"%s\",\n",rank_plan->host_name);
    fprintf(file,"  \"first_layer\": %u,\n",rank_plan->first_layer_index);
    fprintf(file,"  \"layer_count\": %u,\n",rank_plan->layer_count);
    fprintf(file,"  \"quantization\": \"fp8_e4m3\",\n");
    fprintf(file,"  \"fp8_pack_root\": \"%s\",\n",fp8_pack_root != 0 ? fp8_pack_root : "");
    fprintf(file,"  \"listen_port\": %u,\n",rank_plan->listen_port);
    fprintf(file,"  \"next_port\": %u,\n",rank_plan->next_port);
    fprintf(file,"  \"previous_host\": \"%s\",\n",rank_plan->previous_host_name);
    fprintf(file,"  \"next_host\": \"%s\",\n",rank_plan->next_host_name);
    fprintf(file,"  \"input_route\": \"%s\",\n",rank_plan->input_route_name);
    fprintf(file,"  \"output_route\": \"%s\",\n",rank_plan->output_route_name);
    fprintf(file,"  \"max_active_sequence_count\": %u,\n",
        rank_plan->max_active_sequence_count);
    fprintf(file,"  \"hidden_bytes_per_sequence\": %u\n",
        rank_plan->bytes_per_sequence);
    fprintf(file,"}\n");
    if (fclose(file) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13GateGenerateRuntimeFiles(
    const SparkGlm52Pp13GateConfig *configuration,
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan)
{
    char rank_directory[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    char manifest_path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    char pack_list_path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
    SparkStatus status;
    int written;

    if (configuration == 0 || rank_plan == 0 ||
        configuration->generate_missing_runtime_files == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (configuration->generated_root == 0 ||
        configuration->generated_root[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52Pp13GateEnsureDirectory(configuration->generated_root);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52Pp13GateBuildRankDirectory(
        configuration->generated_root,rank_plan->rank_index,
        rank_directory,sizeof(rank_directory));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52Pp13GateEnsureDirectory(rank_directory);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    written = snprintf(
        manifest_path,sizeof(manifest_path),"%s/runtime_manifest.json",
        rank_directory);
    if (written < 0 || (uint32_t)written >= sizeof(manifest_path))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    written = snprintf(
        pack_list_path,sizeof(pack_list_path),"%s/fp8_stage_packs.tsv",
        rank_directory);
    if (written < 0 || (uint32_t)written >= sizeof(pack_list_path))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkGlm52Pp13GateWriteRankManifest(
        rank_plan,configuration->fp8_pack_root,manifest_path);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52Pp13GateWritePackList(
        rank_plan,configuration->fp8_pack_root,pack_list_path);
}

static void SparkGlm52Pp13GatePrintPlan(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    const char *fp8_pack_root,
    SparkStatus pack_status,
    SparkStatus transport_status)
{
    printf("glm52_pp13_rank_gate=1\n");
    printf("rank=%u\n",rank_plan->rank_index);
    printf("host=%s\n",rank_plan->host_name);
    printf("stage=%u:%u\n",
        rank_plan->first_layer_index,rank_plan->layer_count);
    printf("quantization=fp8_e4m3\n");
    printf("fp8_pack_root=%s\n",fp8_pack_root != 0 ? fp8_pack_root : "");
    printf("pack_status=%s\n",SparkStatusToString(pack_status));
    printf("transport_preflight_status=%s\n",SparkStatusToString(transport_status));
    printf("listen_port=%u\n",rank_plan->listen_port);
    printf("next_port=%u\n",rank_plan->next_port);
    printf("previous_host=%s\n",rank_plan->previous_host_name);
    printf("next_host=%s\n",rank_plan->next_host_name);
    printf("input_route=%s\n",rank_plan->input_route_name);
    printf("output_route=%s\n",rank_plan->output_route_name);
    printf("max_active_sequence_count=%u\n",
        rank_plan->max_active_sequence_count);
    printf("hidden_bytes_per_sequence=%u\n",rank_plan->bytes_per_sequence);
    printf("max_packet_bytes=%llu\n",
        (unsigned long long)rank_plan->max_packet_bytes);
}

int main(int argc,char **argv)
{
    SparkGlm52Pp13GateConfig configuration;
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    SparkJsonDocument document;
    SparkStatus status;
    SparkStatus pack_status;
    SparkStatus transport_status;
    SparkStatus generate_status;
    char error_buffer[256];
    int index;

    SparkGlm52Pp13GateInitializeConfig(&configuration);
    SparkJsonDocumentReset(&document);
    for (index = 1; index < argc; ++index)
    {
        if (SparkGlm52Pp13GateApplyArgument(
                &configuration,argc,argv,&index) < 0)
        {
            fprintf(stderr,"invalid argument: %s\n",argv[index]);
            return 2;
        }
    }
    error_buffer[0] = '\0';
    status = SparkGlm52Pp13GateLoadConfigFile(
        &configuration,&document,error_buffer,sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"GLM52 PP13 config error: %s %s\n",
            SparkStatusToString(status),error_buffer);
        SparkJsonDocumentDestroy(&document);
        return 3;
    }
    for (index = 1; index < argc; ++index)
    {
        if (strcmp(argv[index],"--config") == 0)
        {
            index += 1;
            continue;
        }
        if (SparkGlm52Pp13GateApplyArgument(
                &configuration,argc,argv,&index) < 0)
        {
            fprintf(stderr,"invalid argument: %s\n",argv[index]);
            SparkJsonDocumentDestroy(&document);
            return 2;
        }
    }
    if (configuration.rank_is_set == 0u)
    {
        fprintf(stderr,"missing --rank\n");
        SparkJsonDocumentDestroy(&document);
        return 4;
    }
    status = SparkGlm52Pp13RuntimeBuildRankPlan(
        configuration.rank_index,
        configuration.max_active_sequence_count,
        configuration.port_base,
        &rank_plan,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"GLM52 PP13 rank plan failed: %s %s\n",
            SparkStatusToString(status),error_buffer);
        SparkJsonDocumentDestroy(&document);
        return 5;
    }
    pack_status = SPARK_STATUS_OK;
    if (configuration.require_pack_files != 0u)
    {
        pack_status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
            &rank_plan,configuration.fp8_pack_root,error_buffer,
            sizeof(error_buffer));
    }
    transport_status = SPARK_STATUS_OK;
    if (configuration.require_gpudirect_preflight != 0u)
    {
        if ((rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
        {
            transport_status = SparkHiddenTransportGpudirectRdmaVerbsPreflight(
                &rank_plan.output_endpoint,0,0);
        }
        else if ((rank_plan.flags &
                SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
        {
            transport_status = SparkHiddenTransportGpudirectRdmaVerbsPreflight(
                &rank_plan.input_endpoint,0,0);
        }
    }
    SparkGlm52Pp13GatePrintPlan(
        &rank_plan,configuration.fp8_pack_root,pack_status,transport_status);
    SparkJsonDocumentDestroy(&document);
    if (pack_status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"FP8 pack gate failed: %s %s\n",
            SparkStatusToString(pack_status),error_buffer);
        return 6;
    }
    if (transport_status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"GPUDirect transport preflight failed: %s\n",
            SparkStatusToString(transport_status));
        return 7;
    }
    generate_status = SparkGlm52Pp13GateGenerateRuntimeFiles(
        &configuration,&rank_plan);
    printf("generated_runtime_status=%s\n",SparkStatusToString(generate_status));
    if (generate_status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"runtime file generation failed: %s\n",
            SparkStatusToString(generate_status));
        return 8;
    }
    return 0;
}

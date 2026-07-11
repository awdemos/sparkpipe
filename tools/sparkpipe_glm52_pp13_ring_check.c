#define _GNU_SOURCE
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_hidden_transport.h"
#include <cuda_runtime_api.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPARK_RING_CHECK_SEQUENCE_ID 0x52494e47ull
#define SPARK_RING_CHECK_HIDDEN_DIMENSION SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_RING_CHECK_BYTES_PER_SEQUENCE SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES

typedef struct SparkRingCheckConfig
{
    uint32_t rank;
    uint32_t rank_count;
    uint32_t laps;
    uint32_t active_sequence_count;
    uint32_t sideband_bytes_per_sequence;
    uint64_t timeout_ms;
    const char *prev_host;
    const char *next_host;
    const char *transport_path;
    const char *transport_module_id;
    uint32_t transport_capability_flags;
    uint32_t verify_every_hop;
    uint32_t busy_poll;
} SparkRingCheckConfig;

static uint64_t SparkRingCheckMonotonicMs(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC,&now) != 0)
        return 0u;
    return ((uint64_t)now.tv_sec * 1000ull) +
        ((uint64_t)now.tv_nsec / 1000000ull);
}

static uint64_t SparkRingCheckMonotonicNs(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC,&now) != 0)
        return 0u;
    return ((uint64_t)now.tv_sec * 1000000000ull) +
        (uint64_t)now.tv_nsec;
}

static void SparkRingCheckFillPattern(
    uint8_t *buffer,
    uint64_t bytes,
    uint64_t seed)
{
    uint64_t word;
    uint64_t offset;

    word = seed ^ 0x9e3779b97f4a7c15ull;
    for (offset = 0u; offset < bytes; ++offset)
    {
        word ^= word << 13;
        word ^= word >> 7;
        word ^= word << 17;
        buffer[offset] = (uint8_t)word;
    }
}

static SparkStatus SparkRingCheckWaitForPollEvents(
    SparkHiddenTransportSession *session,
    uint64_t deadline_ms)
{
    SparkHiddenTransportPollDescriptor descriptors[4];
    struct pollfd fds[4];
    uint32_t descriptor_count;
    uint32_t descriptor_index;
    uint64_t now_ms;
    int32_t timeout_ms;

    descriptor_count = 0u;
    if (SparkHiddenTransportGetPollDescriptors(
            session,
            descriptors,
            4u,
            &descriptor_count) != SPARK_STATUS_OK ||
        descriptor_count == 0u)
    {
        now_ms = SparkRingCheckMonotonicMs();
        if (now_ms >= deadline_ms)
            return SPARK_STATUS_BUSY;
        (void)poll(0,0,1);
        return SPARK_STATUS_OK;
    }
    for (descriptor_index = 0u;
         descriptor_index < descriptor_count;
         ++descriptor_index)
    {
        memset(&fds[descriptor_index],0,sizeof(fds[descriptor_index]));
        fds[descriptor_index].fd = descriptors[descriptor_index].fd;
        fds[descriptor_index].events =
            (short)descriptors[descriptor_index].events;
    }
    now_ms = SparkRingCheckMonotonicMs();
    if (now_ms >= deadline_ms)
        return SPARK_STATUS_BUSY;
    timeout_ms = (int32_t)(deadline_ms - now_ms);
    if (timeout_ms > 100)
        timeout_ms = 100;
    (void)poll(fds,descriptor_count,timeout_ms);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkRingCheckSendAdapter(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packet)
{
    return SparkHiddenTransportSend(session,packet);
}

static SparkStatus SparkRingCheckPumpUntilOk(
    SparkStatus (*operation)(SparkHiddenTransportSession *,SparkHiddenTransportPacket *),
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packet,
    uint64_t timeout_ms,
    const char *label,
    uint32_t busy_poll)
{
    SparkStatus status;
    uint64_t deadline_ms;

    deadline_ms = SparkRingCheckMonotonicMs() + timeout_ms;
    for (;;)
    {
        status = operation(session,packet);
        if (status == SPARK_STATUS_OK)
            return SPARK_STATUS_OK;
        if (status != SPARK_STATUS_BUSY)
        {
            fprintf(
                stderr,
                "ring_check_%s_failed status=%d lap=%llu\n",
                label,
                (int32_t)status,
                (unsigned long long)packet->token_index);
            return status;
        }
        if (SparkRingCheckMonotonicMs() >= deadline_ms)
        {
            fprintf(
                stderr,
                "ring_check_%s_timeout lap=%llu timeout_ms=%llu\n",
                label,
                (unsigned long long)packet->token_index,
                (unsigned long long)timeout_ms);
            return SPARK_STATUS_BUSY;
        }
        if (busy_poll == 0u)
            (void)SparkRingCheckWaitForPollEvents(session,deadline_ms);
    }
}

static void SparkRingCheckBuildEndpoint(
    const SparkRingCheckConfig *configuration,
    const char *source_host,
    const char *sink_host,
    char *route_buffer,
    uint64_t route_buffer_bytes,
    SparkHiddenTransportEndpoint *endpoint)
{
    memset(endpoint,0,sizeof(*endpoint));
    (void)snprintf(
        route_buffer,
        (size_t)route_buffer_bytes,
        "%s_to_%s_hidden",
        source_host,
        sink_host);
    endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint->capability_flags = configuration->transport_capability_flags;
    endpoint->hidden_dimension = SPARK_RING_CHECK_HIDDEN_DIMENSION;
    endpoint->bytes_per_sequence = SPARK_RING_CHECK_BYTES_PER_SEQUENCE;
    endpoint->max_active_sequence_count =
        configuration->active_sequence_count;
    endpoint->max_packet_bytes =
        (uint64_t)(SPARK_RING_CHECK_BYTES_PER_SEQUENCE +
            configuration->sideband_bytes_per_sequence) *
        (uint64_t)configuration->active_sequence_count;
    endpoint->transport_module_id = configuration->transport_module_id;
    endpoint->route_name = route_buffer;
}

static SparkStatus SparkRingCheckOpenSession(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportDynamicLibrary *transport_library,
    SparkHiddenTransportSession **session,
    const char *label)
{
    SparkStatus status;

    status = SparkHiddenTransportOpen(endpoint,
        &transport_library->transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        session);
    if (status != SPARK_STATUS_OK)
        fprintf(stderr,"ring_check_%s_open_failed status=%d route=%s\n",
            label,(int32_t)status,endpoint->route_name);
    return status;
}

static void SparkRingCheckBuildPacket(
    const SparkRingCheckConfig *configuration,
    uint64_t lap,
    void *hidden_device,
    void *sideband_device,
    void *cuda_stream,
    SparkHiddenTransportPacket *packet)
{
    memset(packet,0,sizeof(*packet));
    packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    packet->active_sequence_count = configuration->active_sequence_count;
    packet->hidden_dimension = SPARK_RING_CHECK_HIDDEN_DIMENSION;
    packet->bytes_per_sequence = SPARK_RING_CHECK_BYTES_PER_SEQUENCE;
    packet->sequence_id = SPARK_RING_CHECK_SEQUENCE_ID;
    packet->token_index = lap;
    packet->hidden_bf16 = hidden_device;
    packet->cuda_stream = cuda_stream;
    if (configuration->sideband_bytes_per_sequence != 0u)
    {
        packet->flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
        packet->sideband_payload = sideband_device;
        packet->sideband_kind =
            SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_INDEXSHARE_SELECTED_TOKENS;
        packet->sideband_bytes_per_sequence =
            configuration->sideband_bytes_per_sequence;
    }
}

static SparkStatus SparkRingCheckVerifyPayload(
    const uint8_t *host_expected,
    const void *device_actual,
    uint64_t bytes,
    uint8_t *host_scratch,
    const char *label,
    uint64_t lap)
{
    if (cudaMemcpy(host_scratch,device_actual,(size_t)bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr,"ring_check_%s_readback_failed lap=%llu\n",
            label,(unsigned long long)lap);
        return SPARK_STATUS_IO_ERROR;
    }
    if (memcmp(host_expected,host_scratch,(size_t)bytes) != 0)
    {
        fprintf(stderr,"ring_check_%s_mismatch lap=%llu bytes=%llu\n",
            label,(unsigned long long)lap,(unsigned long long)bytes);
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    return SPARK_STATUS_OK;
}

static int32_t SparkRingCheckParseArguments(
    int argc,
    char **argv,
    SparkRingCheckConfig *configuration)
{
    int32_t argument_index;

    memset(configuration,0,sizeof(*configuration));
    configuration->rank_count = 13u;
    configuration->laps = 100u;
    configuration->active_sequence_count = 1u;
    configuration->timeout_ms = 30000u;
    configuration->transport_path = "build/libhidden_transport_tcp_cuda.so";
    configuration->transport_module_id = "hidden_transport_tcp_cuda";
    configuration->verify_every_hop = 1u;
    for (argument_index = 1; argument_index < argc; ++argument_index)
    {
        if (strcmp(argv[argument_index],"--rank") == 0 &&
            argument_index + 1 < argc)
            configuration->rank =
                (uint32_t)strtoul(argv[++argument_index],0,10);
        else if (strcmp(argv[argument_index],"--ranks") == 0 &&
            argument_index + 1 < argc)
            configuration->rank_count =
                (uint32_t)strtoul(argv[++argument_index],0,10);
        else if (strcmp(argv[argument_index],"--laps") == 0 &&
            argument_index + 1 < argc)
            configuration->laps =
                (uint32_t)strtoul(argv[++argument_index],0,10);
        else if (strcmp(argv[argument_index],"--active") == 0 &&
            argument_index + 1 < argc)
            configuration->active_sequence_count =
                (uint32_t)strtoul(argv[++argument_index],0,10);
        else if (strcmp(argv[argument_index],"--sideband-bytes") == 0 &&
            argument_index + 1 < argc)
            configuration->sideband_bytes_per_sequence =
                (uint32_t)strtoul(argv[++argument_index],0,10);
        else if (strcmp(argv[argument_index],"--timeout-ms") == 0 &&
            argument_index + 1 < argc)
            configuration->timeout_ms =
                (uint64_t)strtoull(argv[++argument_index],0,10);
        else if (strcmp(argv[argument_index],"--prev") == 0 &&
            argument_index + 1 < argc)
            configuration->prev_host = argv[++argument_index];
        else if (strcmp(argv[argument_index],"--next") == 0 &&
            argument_index + 1 < argc)
            configuration->next_host = argv[++argument_index];
        else if (strcmp(argv[argument_index],"--transport") == 0 &&
            argument_index + 1 < argc)
            configuration->transport_path = argv[++argument_index];
        else if (strcmp(argv[argument_index],"--transport-module") == 0 &&
            argument_index + 1 < argc)
            configuration->transport_module_id = argv[++argument_index];
        else if (strcmp(argv[argument_index],"--end-to-end-verify-only") == 0)
            configuration->verify_every_hop = 0u;
        else if (strcmp(argv[argument_index],"--busy-poll") == 0)
            configuration->busy_poll = 1u;
        else
        {
            fprintf(stderr,"ring_check_bad_argument %s\n",
                argv[argument_index]);
            return -1;
        }
    }
    if (configuration->prev_host == 0 || configuration->next_host == 0 ||
        configuration->rank >= configuration->rank_count ||
        configuration->laps == 0u ||
        configuration->active_sequence_count == 0u)
    {
        fprintf(stderr,
            "usage: --rank N --ranks 13 --prev HOST --next HOST"
            " [--laps K --active A --sideband-bytes S --timeout-ms T"
            " --transport PATH --transport-module ID"
            " --end-to-end-verify-only --busy-poll]\n");
        return -1;
    }
    return 0;
}

int main(int argc,char **argv)
{
    SparkRingCheckConfig configuration;
    SparkHiddenTransportDynamicLibrary transport_library;
    SparkHiddenTransportEndpoint input_endpoint;
    SparkHiddenTransportEndpoint output_endpoint;
    SparkHiddenTransportSession *input_session;
    SparkHiddenTransportSession *output_session;
    SparkHiddenTransportPacket packet;
    SparkStatus status;
    char input_route[128];
    char output_route[128];
    char self_host[32];
    char rank_text[16];
    uint8_t *host_pattern;
    uint8_t *host_scratch;
    void *hidden_device;
    void *hidden_host_mapped;
    void *sideband_device;
    void *sideband_host_mapped;
    cudaStream_t stream;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint64_t lap;
    uint64_t lap_start_ns;
    uint64_t lap_ns;
    uint64_t total_ns;
    uint64_t worst_ns;
    uint64_t best_ns;
    uint64_t hop_start_ns;
    uint64_t hop_ns;
    uint64_t hop_total_ns;
    uint64_t hop_worst_ns;
    uint64_t measured_laps;

    if (SparkRingCheckParseArguments(argc,argv,&configuration) != 0)
        return 1;
    (void)snprintf(rank_text,sizeof(rank_text),"%u",configuration.rank);
    if (setenv("SPARKPIPE_PP13_TRANSPORT_RANK",rank_text,1) != 0)
        return 1;
    if (configuration.rank < 10u)
        (void)snprintf(self_host,sizeof(self_host),"spark%u",
            configuration.rank);
    else
        (void)snprintf(self_host,sizeof(self_host),"spark%c",
            (char)('a' + (configuration.rank - 10u)));
    memset(&transport_library,0,sizeof(transport_library));
    status = SparkHiddenTransportLoadInterfaceFromSharedObject(
        configuration.transport_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        &transport_library);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"ring_check_transport_load_failed status=%d path=%s\n",
            (int32_t)status,configuration.transport_path);
        return 1;
    }
    configuration.transport_capability_flags =
        transport_library.transport_interface.capability_flags;
    SparkRingCheckBuildEndpoint(&configuration,configuration.prev_host,
        self_host,input_route,sizeof(input_route),&input_endpoint);
    SparkRingCheckBuildEndpoint(&configuration,self_host,
        configuration.next_host,output_route,sizeof(output_route),
        &output_endpoint);
    input_session = 0;
    output_session = 0;
    if ((configuration.rank & 1u) == 0u)
    {
        status = SparkRingCheckOpenSession(&input_endpoint,&transport_library,
            &input_session,"input");
        if (status == SPARK_STATUS_OK)
            status = SparkRingCheckOpenSession(&output_endpoint,
                &transport_library,&output_session,"output");
    }
    else
    {
        status = SparkRingCheckOpenSession(&output_endpoint,&transport_library,
            &output_session,"output");
        if (status == SPARK_STATUS_OK)
            status = SparkRingCheckOpenSession(&input_endpoint,
                &transport_library,&input_session,"input");
    }
    if (status != SPARK_STATUS_OK)
        return 1;
    hidden_bytes = (uint64_t)SPARK_RING_CHECK_BYTES_PER_SEQUENCE *
        (uint64_t)configuration.active_sequence_count;
    sideband_bytes = (uint64_t)configuration.sideband_bytes_per_sequence *
        (uint64_t)configuration.active_sequence_count;
    host_pattern = 0;
    host_scratch = 0;
    if (cudaHostAlloc((void **)&host_pattern,
            (size_t)(hidden_bytes + sideband_bytes),
            cudaHostAllocDefault) != cudaSuccess)
        host_pattern = 0;
    if (cudaHostAlloc((void **)&host_scratch,
            (size_t)(hidden_bytes + sideband_bytes),
            cudaHostAllocDefault) != cudaSuccess)
        host_scratch = 0;
    hidden_device = 0;
    hidden_host_mapped = 0;
    sideband_device = 0;
    sideband_host_mapped = 0;
    stream = 0;
    if ((configuration.transport_capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY) != 0u)
    {
        if (cudaHostAlloc(&hidden_host_mapped,(size_t)hidden_bytes,
                cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess)
            (void)cudaHostGetDevicePointer(&hidden_device,hidden_host_mapped,0);
        if (sideband_bytes != 0u &&
            cudaHostAlloc(&sideband_host_mapped,(size_t)sideband_bytes,
                cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess)
            (void)cudaHostGetDevicePointer(&sideband_device,
                sideband_host_mapped,0);
    }
    else
    {
        (void)cudaMalloc(&hidden_device,(size_t)hidden_bytes);
        if (sideband_bytes != 0u)
            (void)cudaMalloc(&sideband_device,(size_t)sideband_bytes);
    }
    if (host_pattern == 0 || host_scratch == 0 ||
        cudaStreamCreate(&stream) != cudaSuccess || hidden_device == 0 ||
        (sideband_bytes != 0u && sideband_device == 0))
    {
        fprintf(stderr,"ring_check_alloc_failed hidden=%llu sideband=%llu\n",
            (unsigned long long)hidden_bytes,
            (unsigned long long)sideband_bytes);
        return 1;
    }
    printf("ring_check_ready rank=%u self=%s prev=%s next=%s laps=%u"
        " active=%u sideband=%u transport=%s mapped=%u verify_every_hop=%u"
        " busy_poll=%u\n",
        configuration.rank,self_host,configuration.prev_host,
        configuration.next_host,configuration.laps,
        configuration.active_sequence_count,
        configuration.sideband_bytes_per_sequence,
        configuration.transport_module_id,
        hidden_host_mapped != 0 ? 1u : 0u,
        configuration.verify_every_hop,
        configuration.busy_poll);
    fflush(stdout);
    total_ns = 0u;
    worst_ns = 0u;
    best_ns = ~0ull;
    hop_start_ns = 0u;
    hop_ns = 0u;
    hop_total_ns = 0u;
    hop_worst_ns = 0u;
    measured_laps = configuration.laps > 1u ?
        (uint64_t)configuration.laps - 1u : 1u;
    for (lap = 0u; lap < configuration.laps; ++lap)
    {
        if (configuration.rank == 0u ||
            configuration.verify_every_hop != 0u)
            SparkRingCheckFillPattern(host_pattern,
                hidden_bytes + sideband_bytes,
                lap * (uint64_t)configuration.rank_count);
        if (configuration.rank == 0u)
        {
            if (cudaMemcpy(hidden_device,host_pattern,(size_t)hidden_bytes,
                    cudaMemcpyHostToDevice) != cudaSuccess ||
                (sideband_bytes != 0u &&
                 cudaMemcpy(sideband_device,host_pattern + hidden_bytes,
                     (size_t)sideband_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess))
            {
                fprintf(stderr,"ring_check_upload_failed lap=%llu\n",
                    (unsigned long long)lap);
                return 1;
            }
            SparkRingCheckBuildPacket(&configuration,lap,hidden_device,
                sideband_device,stream,&packet);
            lap_start_ns = SparkRingCheckMonotonicNs();
            status = SparkRingCheckPumpUntilOk(SparkRingCheckSendAdapter,
                output_session,&packet,configuration.timeout_ms,"send",
                configuration.busy_poll);
            if (status != SPARK_STATUS_OK)
                return 1;
            SparkRingCheckBuildPacket(&configuration,lap,hidden_device,
                sideband_device,stream,&packet);
            status = SparkRingCheckPumpUntilOk(SparkHiddenTransportPostReceive,
                input_session,&packet,configuration.timeout_ms,"receive",
                configuration.busy_poll);
            if (status != SPARK_STATUS_OK)
                return 1;
            lap_ns = SparkRingCheckMonotonicNs() - lap_start_ns;
            if (SparkRingCheckVerifyPayload(host_pattern,hidden_device,
                    hidden_bytes,host_scratch,"hidden",lap) !=
                SPARK_STATUS_OK)
                return 1;
            if (sideband_bytes != 0u &&
                SparkRingCheckVerifyPayload(host_pattern + hidden_bytes,
                    sideband_device,sideband_bytes,host_scratch,"sideband",
                    lap) != SPARK_STATUS_OK)
                return 1;
            if (lap != 0u || configuration.laps == 1u)
            {
                total_ns += lap_ns;
                if (lap_ns > worst_ns)
                    worst_ns = lap_ns;
                if (lap_ns < best_ns)
                    best_ns = lap_ns;
            }
            printf("ring_check_lap lap=%llu us=%llu\n",
                (unsigned long long)lap,
                (unsigned long long)(lap_ns / 1000ull));
            fflush(stdout);
        }
        else
        {
            SparkRingCheckBuildPacket(&configuration,lap,hidden_device,
                sideband_device,stream,&packet);
            status = SparkRingCheckPumpUntilOk(SparkHiddenTransportPostReceive,
                input_session,&packet,configuration.timeout_ms,"receive",
                configuration.busy_poll);
            if (status != SPARK_STATUS_OK)
                return 1;
            hop_start_ns = SparkRingCheckMonotonicNs();
            if (configuration.verify_every_hop != 0u)
            {
                if (SparkRingCheckVerifyPayload(host_pattern,hidden_device,
                        hidden_bytes,host_scratch,"hidden",lap) !=
                    SPARK_STATUS_OK)
                    return 1;
                if (sideband_bytes != 0u &&
                    SparkRingCheckVerifyPayload(host_pattern + hidden_bytes,
                        sideband_device,sideband_bytes,host_scratch,"sideband",
                        lap) != SPARK_STATUS_OK)
                    return 1;
            }
            SparkRingCheckBuildPacket(&configuration,lap,hidden_device,
                sideband_device,stream,&packet);
            status = SparkRingCheckPumpUntilOk(SparkRingCheckSendAdapter,
                output_session,&packet,configuration.timeout_ms,"send",
                configuration.busy_poll);
            if (status != SPARK_STATUS_OK)
                return 1;
            hop_ns = SparkRingCheckMonotonicNs() - hop_start_ns;
            if (lap != 0u || configuration.laps == 1u)
            {
                hop_total_ns += hop_ns;
                if (hop_ns > hop_worst_ns)
                    hop_worst_ns = hop_ns;
            }
        }
    }
    if (configuration.rank == 0u)
        printf("ring_check_result laps=%u measured_laps=%llu avg_us=%llu"
            " best_us=%llu worst_us=%llu\n",
            configuration.laps,
            (unsigned long long)measured_laps,
            (unsigned long long)(total_ns / measured_laps / 1000ull),
            (unsigned long long)(best_ns / 1000ull),
            (unsigned long long)(worst_ns / 1000ull));
    else
        printf("ring_check_forwarder_done rank=%u laps=%u hop_avg_us=%llu"
            " hop_worst_ms=%llu\n",
            configuration.rank,configuration.laps,
            (unsigned long long)(hop_total_ns / measured_laps / 1000ull),
            (unsigned long long)(hop_worst_ns / 1000000ull));
    SparkHiddenTransportClose(input_session);
    SparkHiddenTransportClose(output_session);
    (void)cudaStreamDestroy(stream);
    return 0;
}

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_hidden_transport.h"

#define SPARK_HIDDEN_TCP_CUDA_MAGIC 0x48544355u
#define SPARK_HIDDEN_TCP_CUDA_HOST_BYTES 32u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_BASE 52100u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_OFFSET 1000u
#define SPARK_HIDDEN_TCP_CUDA_CONNECT_ATTEMPTS 600u
#define SPARK_HIDDEN_TCP_CUDA_CONNECT_SLEEP_US 100000u

typedef struct SparkHiddenTcpCudaHeader
{
    uint32_t magic;
    uint32_t active_sequence_count;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
} SparkHiddenTcpCudaHeader;

typedef struct SparkHiddenTcpCudaState
{
    SparkHiddenTransportEndpoint endpoint;
    int32_t local_rank;
    int32_t source_rank;
    int32_t sink_rank;
    uint32_t port_base;
    uint32_t is_sender;
    int listen_fd;
    int socket_fd;
    uint8_t *host_buffer;
    uint64_t host_buffer_bytes;
    SparkHiddenTransportCompletion completion;
    uint32_t completion_ready;
    char source_host[SPARK_HIDDEN_TCP_CUDA_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_TCP_CUDA_HOST_BYTES];
} SparkHiddenTcpCudaState;

static int32_t SparkHiddenTcpCudaRankFromHost(const char *host)
{
    if (host == 0 || host[0] != 's' || host[1] != 'p' || host[2] != 'a' ||
        host[3] != 'r' || host[4] != 'k' || host[5] == '\0')
        return -1;
    if (host[5] >= '0' && host[5] <= '9' && host[6] == '\0')
        return (int32_t)(host[5] - '0');
    if (host[5] >= 'a' && host[5] <= 'c' && host[6] == '\0')
        return (int32_t)(10 + (host[5] - 'a'));
    return -1;
}

static uint32_t SparkHiddenTcpCudaParseUintEnv(const char *name,uint32_t fallback)
{
    const char *value;
    char *end;
    unsigned long parsed;
    value = getenv(name);
    if (value == 0 || value[0] == '\0')
        return fallback;
    parsed = strtoul(value,&end,10);
    if (end == value || *end != '\0' || parsed > 65535ul)
        return fallback;
    return (uint32_t)parsed;
}

static SparkStatus SparkHiddenTcpCudaParseRoute(
    const char *route_name,
    char *source_host,
    char *sink_host)
{
    const char *middle;
    const char *suffix;
    uint64_t source_bytes;
    uint64_t sink_bytes;
    if (route_name == 0 || source_host == 0 || sink_host == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    middle = strstr(route_name,"_to_");
    suffix = strstr(route_name,"_hidden");
    if (middle == 0 || suffix == 0 || middle >= suffix)
        return SPARK_STATUS_INVALID_ARGUMENT;
    source_bytes = (uint64_t)(middle - route_name);
    sink_bytes = (uint64_t)(suffix - (middle + 4));
    if (source_bytes == 0u || sink_bytes == 0u ||
        source_bytes >= SPARK_HIDDEN_TCP_CUDA_HOST_BYTES ||
        sink_bytes >= SPARK_HIDDEN_TCP_CUDA_HOST_BYTES)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    memcpy(source_host,route_name,(size_t)source_bytes);
    source_host[source_bytes] = '\0';
    memcpy(sink_host,middle + 4,(size_t)sink_bytes);
    sink_host[sink_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static int SparkHiddenTcpCudaCloseFd(int fd)
{
    if (fd >= 0)
        close(fd);
    return -1;
}

static int SparkHiddenTcpCudaListen(uint32_t port)
{
    struct sockaddr_in address;
    int fd;
    int value;
    fd = socket(AF_INET,SOCK_STREAM,0);
    if (fd < 0)
        return -1;
    value = 1;
    setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&value,sizeof(value));
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((uint16_t)port);
    if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0 ||
        listen(fd,16) < 0)
        return SparkHiddenTcpCudaCloseFd(fd);
    return fd;
}

static int SparkHiddenTcpCudaConnectOnce(const char *host,uint32_t port)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16];
    int fd;
    fd = -1;
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_text,sizeof(port_text),"%u",port);
    if (getaddrinfo(host,port_text,&hints,&result) != 0)
        return -1;
    for (entry = result; entry != 0; entry = entry->ai_next)
    {
        fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
        if (fd < 0)
            continue;
        if (connect(fd,entry->ai_addr,entry->ai_addrlen) == 0)
            break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    return fd;
}

static int SparkHiddenTcpCudaConnect(const char *host,uint32_t port)
{
    uint32_t attempt;
    int fd;
    for (attempt = 0u; attempt < SPARK_HIDDEN_TCP_CUDA_CONNECT_ATTEMPTS; ++attempt)
    {
        fd = SparkHiddenTcpCudaConnectOnce(host,port);
        if (fd >= 0)
            return fd;
        usleep(SPARK_HIDDEN_TCP_CUDA_CONNECT_SLEEP_US);
    }
    return -1;
}

static int32_t SparkHiddenTcpCudaWriteAll(int fd,const void *buffer,uint64_t bytes)
{
    const uint8_t *ptr;
    ssize_t wrote;
    ptr = (const uint8_t *)buffer;
    while (bytes != 0u)
    {
        wrote = write(fd,ptr,(size_t)bytes);
        if (wrote < 0 && errno == EINTR)
            continue;
        if (wrote <= 0)
            return -1;
        ptr += wrote;
        bytes -= (uint64_t)wrote;
    }
    return 0;
}

static int32_t SparkHiddenTcpCudaReadAll(int fd,void *buffer,uint64_t bytes)
{
    uint8_t *ptr;
    ssize_t got;
    ptr = (uint8_t *)buffer;
    while (bytes != 0u)
    {
        got = read(fd,ptr,(size_t)bytes);
        if (got < 0 && errno == EINTR)
            continue;
        if (got <= 0)
            return -1;
        ptr += got;
        bytes -= (uint64_t)got;
    }
    return 0;
}

static SparkStatus SparkHiddenTcpCudaEnsureSocket(SparkHiddenTcpCudaState *state)
{
    const char *host;
    if (state->socket_fd >= 0)
        return SPARK_STATUS_OK;
    if (state->is_sender != 0u)
    {
        host = getenv("SPARKPIPE_PP13_TRANSPORT_HOST_OVERRIDE");
        if (host == 0 || host[0] == '\0')
            host = state->sink_host;
        state->socket_fd = SparkHiddenTcpCudaConnect(
            host,
            state->port_base + (uint32_t)state->sink_rank);
    }
    else
    {
        state->socket_fd = accept(state->listen_fd,0,0);
    }
    return state->socket_fd >= 0 ? SPARK_STATUS_OK : SPARK_STATUS_ROUTE_NOT_FOUND;
}

static SparkStatus SparkHiddenTcpCudaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenTcpCudaState *state;
    const char *rank_text;
    uint64_t sideband_capacity;
    if (endpoint == 0 || transport_state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state = (SparkHiddenTcpCudaState *)calloc(1u,sizeof(*state));
    if (state == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    state->endpoint = *endpoint;
    state->listen_fd = -1;
    state->socket_fd = -1;
    state->port_base = SparkHiddenTcpCudaParseUintEnv(
        "SPARKPIPE_PP13_HIDDEN_TRANSPORT_PORT_BASE",
        SparkHiddenTcpCudaParseUintEnv(
            "SPARKPIPE_PP13_TRANSPORT_PORT_BASE",
            SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_BASE) +
            SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_OFFSET);
    rank_text = getenv("SPARKPIPE_PP13_TRANSPORT_RANK");
    if (rank_text == 0 || rank_text[0] == '\0')
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->local_rank = (int32_t)SparkHiddenTcpCudaParseUintEnv(
        "SPARKPIPE_PP13_TRANSPORT_RANK",
        1000u);
    if (SparkHiddenTcpCudaParseRoute(endpoint->route_name,
            state->source_host,state->sink_host) != SPARK_STATUS_OK)
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->source_rank = SparkHiddenTcpCudaRankFromHost(state->source_host);
    state->sink_rank = SparkHiddenTcpCudaRankFromHost(state->sink_host);
    if (state->source_rank < 0 || state->sink_rank < 0)
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->local_rank == state->source_rank)
        state->is_sender = 1u;
    else if (state->local_rank == state->sink_rank)
        state->is_sender = 0u;
    else
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->is_sender == 0u)
    {
        state->listen_fd = SparkHiddenTcpCudaListen(
            state->port_base + (uint32_t)state->sink_rank);
        if (state->listen_fd < 0)
        {
            free(state);
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
    }
    sideband_capacity = (uint64_t)SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH;
    sideband_capacity *= sizeof(uint32_t);
    state->host_buffer_bytes = sizeof(SparkHiddenTcpCudaHeader) +
        endpoint->max_packet_bytes + sideband_capacity;
    state->host_buffer = (uint8_t *)malloc((size_t)state->host_buffer_bytes);
    if (state->host_buffer == 0)
    {
        if (state->listen_fd >= 0)
            close(state->listen_fd);
        free(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenTcpCudaDestroy(void *transport_state)
{
    SparkHiddenTcpCudaState *state;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0)
        return;
    if (state->socket_fd >= 0)
        close(state->socket_fd);
    if (state->listen_fd >= 0)
        close(state->listen_fd);
    free(state->host_buffer);
    free(state);
}

static SparkStatus SparkHiddenTcpCudaSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaState *state;
    SparkHiddenTcpCudaHeader header;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint64_t offset;
    SparkStatus status;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenTcpCudaEnsureSocket(state);
    if (status != SPARK_STATUS_OK)
        return status;
    hidden_bytes = (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    sideband_bytes = (uint64_t)packet->sideband_bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    if (sizeof(header) + hidden_bytes + sideband_bytes >
        state->host_buffer_bytes)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    memset(&header,0,sizeof(header));
    header.magic = SPARK_HIDDEN_TCP_CUDA_MAGIC;
    header.active_sequence_count = packet->active_sequence_count;
    header.hidden_dimension = packet->hidden_dimension;
    header.bytes_per_sequence = packet->bytes_per_sequence;
    header.sequence_id = packet->sequence_id;
    header.token_index = packet->token_index;
    header.sideband_kind = packet->sideband_kind;
    header.sideband_bytes_per_sequence = packet->sideband_bytes_per_sequence;
    memcpy(state->host_buffer,&header,sizeof(header));
    offset = sizeof(header);
    if (cudaMemcpy(state->host_buffer + offset,packet->hidden_bf16,
            (size_t)hidden_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    offset += hidden_bytes;
    if (sideband_bytes != 0u &&
        cudaMemcpy(state->host_buffer + offset,packet->sideband_payload,
            (size_t)sideband_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (SparkHiddenTcpCudaWriteAll(state->socket_fd,state->host_buffer,
            sizeof(header) + hidden_bytes + sideband_bytes) < 0)
    {
        close(state->socket_fd);
        state->socket_fd = -1;
        return SPARK_STATUS_IO_ERROR;
    }
    memset(&state->completion,0,sizeof(state->completion));
    state->completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    state->completion.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    state->completion.status = SPARK_STATUS_OK;
    state->completion.active_sequence_count = packet->active_sequence_count;
    state->completion.sequence_id = packet->sequence_id;
    state->completion.token_index = packet->token_index;
    state->completion.transfer_bytes = hidden_bytes + sideband_bytes;
    state->completion_ready = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaState *state;
    SparkHiddenTcpCudaHeader header;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    SparkStatus status;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenTcpCudaEnsureSocket(state);
    if (status != SPARK_STATUS_OK)
        return status;
    if (SparkHiddenTcpCudaReadAll(state->socket_fd,&header,sizeof(header)) < 0)
        return SPARK_STATUS_IO_ERROR;
    hidden_bytes = (uint64_t)header.bytes_per_sequence *
        (uint64_t)header.active_sequence_count;
    sideband_bytes = (uint64_t)header.sideband_bytes_per_sequence *
        (uint64_t)header.active_sequence_count;
    if (header.magic != SPARK_HIDDEN_TCP_CUDA_MAGIC ||
        header.active_sequence_count != packet->active_sequence_count ||
        header.hidden_dimension != packet->hidden_dimension ||
        header.bytes_per_sequence != packet->bytes_per_sequence ||
        header.sequence_id != packet->sequence_id ||
        header.token_index != packet->token_index ||
        header.sideband_kind != packet->sideband_kind ||
        header.sideband_bytes_per_sequence != packet->sideband_bytes_per_sequence ||
        sizeof(header) + hidden_bytes + sideband_bytes > state->host_buffer_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (SparkHiddenTcpCudaReadAll(state->socket_fd,state->host_buffer,
            hidden_bytes + sideband_bytes) < 0)
        return SPARK_STATUS_IO_ERROR;
    if (cudaMemcpyAsync((void *)packet->hidden_bf16,state->host_buffer,
            (size_t)hidden_bytes,cudaMemcpyHostToDevice,
            (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (sideband_bytes != 0u &&
        cudaMemcpyAsync((void *)packet->sideband_payload,
            state->host_buffer + hidden_bytes,(size_t)sideband_bytes,
            cudaMemcpyHostToDevice,
            (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    memset(&state->completion,0,sizeof(state->completion));
    state->completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    state->completion.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    state->completion.status = SPARK_STATUS_OK;
    state->completion.active_sequence_count = packet->active_sequence_count;
    state->completion.sequence_id = packet->sequence_id;
    state->completion.token_index = packet->token_index;
    state->completion.transfer_bytes = hidden_bytes + sideband_bytes;
    state->completion_ready = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    SparkHiddenTcpCudaState *state;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || completion == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->completion_ready == 0u)
    {
        memset(completion,0,sizeof(*completion));
        completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
        completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
        completion->status = SPARK_STATUS_BUSY;
        return SPARK_STATUS_OK;
    }
    *completion = state->completion;
    state->completion_ready = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;
    uint32_t packet_index;
    if (packets == 0 || packet_count == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenTcpCudaSend(transport_state,&packets[packet_index]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaPostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;
    uint32_t packet_index;
    if (packets == 0 || packet_count == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenTcpCudaPostReceive(
            transport_state,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface transport_interface;
    memset(&transport_interface,0,sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PIPELINE_HOST_STAGED_CAPS;
    transport_interface.initialize = SparkHiddenTcpCudaInitialize;
    transport_interface.destroy = SparkHiddenTcpCudaDestroy;
    transport_interface.post_receive = SparkHiddenTcpCudaPostReceive;
    transport_interface.send = SparkHiddenTcpCudaSend;
    transport_interface.poll = SparkHiddenTcpCudaPoll;
    transport_interface.post_receive_batch = SparkHiddenTcpCudaPostReceiveBatch;
    transport_interface.send_batch = SparkHiddenTcpCudaSendBatch;
    return &transport_interface;
}

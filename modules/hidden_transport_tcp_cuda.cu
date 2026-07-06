#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_hidden_transport.h"

#define SPARK_HIDDEN_TCP_CUDA_MAGIC 0x48544355u
#define SPARK_HIDDEN_TCP_CUDA_HOST_BYTES 32u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_BASE 52100u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PORT_OFFSET 1000u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PENDING_DEPTH 2048u
#define SPARK_HIDDEN_TCP_CUDA_DEFAULT_PENDING_HASH_SLOTS 4096u
#define SPARK_HIDDEN_TCP_CUDA_NO_PENDING UINT32_MAX

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

typedef struct SparkHiddenTcpCudaPendingPacket
{
    uint32_t active;
    uint32_t next_pending_index;
    uint32_t next_free_index;
    SparkHiddenTcpCudaHeader header;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint8_t *payload;
} SparkHiddenTcpCudaPendingPacket;

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
    uint32_t socket_connecting;
    uint8_t *host_buffer;
    uint64_t host_buffer_bytes;
    SparkHiddenTcpCudaPendingPacket *pending_packets;
    uint32_t *pending_hash_heads;
    uint32_t pending_hash_slots;
    uint32_t pending_depth;
    uint32_t free_pending_head;
    uint64_t pending_payload_bytes;
    SparkHiddenTransportCompletion completion;
    uint32_t completion_ready;
    SparkHiddenTcpCudaHeader incoming_header;
    uint64_t incoming_header_offset;
    uint64_t incoming_payload_offset;
    uint64_t incoming_payload_bytes;
    uint64_t incoming_hidden_bytes;
    uint64_t incoming_sideband_bytes;
    uint32_t incoming_state;
    uint32_t incoming_matches_request;
    uint32_t incoming_pending_index;
    uint8_t *incoming_payload;
    SparkHiddenTcpCudaHeader outgoing_header;
    uint64_t outgoing_bytes;
    uint64_t outgoing_offset;
    uint32_t outgoing_active;
    uint32_t debug_enabled;
    char source_host[SPARK_HIDDEN_TCP_CUDA_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_TCP_CUDA_HOST_BYTES];
} SparkHiddenTcpCudaState;

static int32_t SparkHiddenTcpCudaRankFromHost(const char *host)
{
    uint32_t tail;
    char extra;

    if (host == 0 || host[0] != 's' || host[1] != 'p' || host[2] != 'a' ||
        host[3] != 'r' || host[4] != 'k' || host[5] == '\0')
    {
        extra = '\0';
        if (sscanf(host,"10.10.100.%u%c",&tail,&extra) == 1 &&
            tail >= 10u && tail <= 22u)
            return (int32_t)(tail - 10u);
        return -1;
    }
    if (host[5] >= '0' && host[5] <= '9' && host[6] == '\0')
        return (int32_t)(host[5] - '0');
    if (host[5] >= 'a' && host[5] <= 'c' && host[6] == '\0')
        return (int32_t)(10 + (host[5] - 'a'));
    return -1;
}

static uint32_t SparkHiddenTcpCudaHeaderHash(
    const SparkHiddenTcpCudaHeader *header)
{
    uint64_t hash;
    if (header == 0)
        return 0u;
    hash = header->sequence_id;
    hash ^= (header->token_index * 0x9e3779b97f4a7c15ull);
    hash ^= ((uint64_t)header->active_sequence_count << 32);
    hash ^= ((uint64_t)header->sideband_kind << 16);
    return (uint32_t)(hash ^ (hash >> 32));
}

static uint64_t SparkHiddenTcpCudaPayloadHash(
    const uint8_t *payload,
    uint64_t payload_bytes)
{
    uint64_t hash;
    uint64_t index;
    if (payload == 0)
        return 0u;
    hash = 1469598103934665603ull;
    for (index = 0u; index < payload_bytes; ++index)
    {
        hash ^= (uint64_t)payload[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint32_t SparkHiddenTcpCudaPacketHash(
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaHeader header;
    if (packet == 0)
        return 0u;
    memset(&header,0,sizeof(header));
    header.active_sequence_count = packet->active_sequence_count;
    header.sequence_id = packet->sequence_id;
    header.token_index = packet->token_index;
    header.sideband_kind = packet->sideband_kind;
    return SparkHiddenTcpCudaHeaderHash(&header);
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

static uint32_t SparkHiddenTcpCudaDebugEnabled(void)
{
    const char *value;
    value = getenv("SPARKPIPE_HIDDEN_TCP_DEBUG");
    return value != 0 && value[0] != '\0' && strcmp(value,"0") != 0;
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

static void SparkHiddenTcpCudaCloseSocket(SparkHiddenTcpCudaState *state)
{
    if (state == 0)
        return;
    if (state->socket_fd >= 0)
        close(state->socket_fd);
    state->socket_fd = -1;
    state->socket_connecting = 0u;
}

static int32_t SparkHiddenTcpCudaSetNonblocking(int fd)
{
    int flags;
    if (fd < 0)
        return -1;
    flags = fcntl(fd,F_GETFL,0);
    if (flags < 0)
        return -1;
    if (fcntl(fd,F_SETFL,flags | O_NONBLOCK) < 0)
        return -1;
    return 0;
}

static void SparkHiddenTcpCudaConfigureSocket(int fd)
{
    int value;
    if (fd < 0)
        return;
    value = 1;
    (void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&value,sizeof(value));
    (void)setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&value,sizeof(value));
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
        listen(fd,16) < 0 ||
        SparkHiddenTcpCudaSetNonblocking(fd) < 0)
        return SparkHiddenTcpCudaCloseFd(fd);
    return fd;
}

static SparkStatus SparkHiddenTcpCudaFinishConnect(SparkHiddenTcpCudaState *state)
{
    socklen_t error_bytes;
    int error;
    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->socket_fd < 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    if (state->socket_connecting == 0u)
        return SPARK_STATUS_OK;
    error = 0;
    error_bytes = sizeof(error);
    if (getsockopt(state->socket_fd,SOL_SOCKET,SO_ERROR,
            &error,&error_bytes) < 0)
    {
        SparkHiddenTcpCudaCloseSocket(state);
        return SPARK_STATUS_BUSY;
    }
    if (error == 0)
    {
        state->socket_connecting = 0u;
        SparkHiddenTcpCudaConfigureSocket(state->socket_fd);
        return SPARK_STATUS_OK;
    }
    if (error == EINPROGRESS || error == EALREADY ||
        error == EWOULDBLOCK || error == EAGAIN)
        return SPARK_STATUS_BUSY;
    SparkHiddenTcpCudaCloseSocket(state);
    return SPARK_STATUS_BUSY;
}

static SparkStatus SparkHiddenTcpCudaBeginConnect(
    SparkHiddenTcpCudaState *state,
    const char *host,
    uint32_t port)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16];
    SparkStatus status;
    int fd;
    int connect_result;
    if (state == 0 || host == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->socket_fd >= 0)
        return SparkHiddenTcpCudaFinishConnect(state);
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_text,sizeof(port_text),"%u",port);
    if (getaddrinfo(host,port_text,&hints,&result) != 0)
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    status = SPARK_STATUS_ROUTE_NOT_FOUND;
    for (entry = result; entry != 0; entry = entry->ai_next)
    {
        fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
        if (fd < 0)
            continue;
        if (SparkHiddenTcpCudaSetNonblocking(fd) < 0)
        {
            close(fd);
            continue;
        }
        SparkHiddenTcpCudaConfigureSocket(fd);
        connect_result = connect(fd,entry->ai_addr,entry->ai_addrlen);
        if (connect_result == 0)
        {
            state->socket_fd = fd;
            state->socket_connecting = 0u;
            status = SPARK_STATUS_OK;
            break;
        }
        if (errno == EINPROGRESS || errno == EALREADY ||
            errno == EWOULDBLOCK || errno == EAGAIN)
        {
            state->socket_fd = fd;
            state->socket_connecting = 1u;
            status = SPARK_STATUS_BUSY;
            break;
        }
        close(fd);
    }
    freeaddrinfo(result);
    return status;
}

static SparkStatus SparkHiddenTcpCudaReadSome(
    int fd,
    void *buffer,
    uint64_t bytes,
    uint64_t *bytes_read)
{
    ssize_t got;
    if (fd < 0 || buffer == 0 || bytes == 0u || bytes_read == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *bytes_read = 0u;
    for (;;)
    {
        got = recv(fd,buffer,(size_t)bytes,MSG_DONTWAIT);
        if (got < 0 && errno == EINTR)
            continue;
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return SPARK_STATUS_BUSY;
        if (got <= 0)
            return SPARK_STATUS_IO_ERROR;
        *bytes_read = (uint64_t)got;
        return SPARK_STATUS_OK;
    }
}

static SparkStatus SparkHiddenTcpCudaWriteSome(
    int fd,
    const void *buffer,
    uint64_t bytes,
    uint64_t *bytes_written)
{
    ssize_t wrote;
    if (fd < 0 || buffer == 0 || bytes == 0u || bytes_written == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *bytes_written = 0u;
    for (;;)
    {
        wrote = send(fd,buffer,(size_t)bytes,MSG_DONTWAIT);
        if (wrote < 0 && errno == EINTR)
            continue;
        if (wrote < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return SPARK_STATUS_BUSY;
        if (wrote <= 0)
            return SPARK_STATUS_IO_ERROR;
        *bytes_written = (uint64_t)wrote;
        return SPARK_STATUS_OK;
    }
}

static uint32_t SparkHiddenTcpCudaHeaderMatchesPacket(
    const SparkHiddenTcpCudaHeader *header,
    const SparkHiddenTransportPacket *packet)
{
    if (header == 0 || packet == 0)
        return 0u;
    return header->magic == SPARK_HIDDEN_TCP_CUDA_MAGIC &&
        header->active_sequence_count == packet->active_sequence_count &&
        header->hidden_dimension == packet->hidden_dimension &&
        header->bytes_per_sequence == packet->bytes_per_sequence &&
        header->sequence_id == packet->sequence_id &&
        header->token_index == packet->token_index &&
        header->sideband_kind == packet->sideband_kind &&
        header->sideband_bytes_per_sequence ==
            packet->sideband_bytes_per_sequence;
}

static SparkStatus SparkHiddenTcpCudaValidateIncomingHeader(
    const SparkHiddenTcpCudaState *state,
    const SparkHiddenTcpCudaHeader *header,
    uint64_t *hidden_bytes_out,
    uint64_t *sideband_bytes_out)
{
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint64_t payload_bytes;
    if (state == 0 || header == 0 ||
        hidden_bytes_out == 0 || sideband_bytes_out == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    hidden_bytes = (uint64_t)header->bytes_per_sequence *
        (uint64_t)header->active_sequence_count;
    sideband_bytes = (uint64_t)header->sideband_bytes_per_sequence *
        (uint64_t)header->active_sequence_count;
    payload_bytes = hidden_bytes + sideband_bytes;
    if (header->magic != SPARK_HIDDEN_TCP_CUDA_MAGIC ||
        header->active_sequence_count == 0u ||
        header->active_sequence_count > state->endpoint.max_active_sequence_count ||
        header->hidden_dimension != state->endpoint.hidden_dimension ||
        header->bytes_per_sequence != state->endpoint.bytes_per_sequence ||
        hidden_bytes / header->active_sequence_count !=
            (uint64_t)header->bytes_per_sequence ||
        sideband_bytes / header->active_sequence_count !=
            (uint64_t)header->sideband_bytes_per_sequence ||
        payload_bytes < hidden_bytes ||
        payload_bytes > state->pending_payload_bytes ||
        payload_bytes > state->endpoint.max_packet_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *hidden_bytes_out = hidden_bytes;
    *sideband_bytes_out = sideband_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaCopyPayloadToPacket(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTransportPacket *packet,
    const uint8_t *payload,
    uint64_t hidden_bytes,
    uint64_t sideband_bytes)
{
    uint64_t hidden_hash;
    uint64_t sideband_hash;
    if (state == 0 || packet == 0 || payload == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (cudaMemcpyAsync((void *)packet->hidden_bf16,payload,
            (size_t)hidden_bytes,cudaMemcpyHostToDevice,
            (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (sideband_bytes != 0u &&
        cudaMemcpyAsync((void *)packet->sideband_payload,
            payload + hidden_bytes,(size_t)sideband_bytes,
            cudaMemcpyHostToDevice,
            (cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
        return SPARK_STATUS_IO_ERROR;
    if (state->debug_enabled != 0u)
    {
        hidden_hash = SparkHiddenTcpCudaPayloadHash(payload,hidden_bytes);
        sideband_hash = SparkHiddenTcpCudaPayloadHash(
            payload + hidden_bytes,
            sideband_bytes);
        fprintf(stderr,
            "hidden_tcp_deliver seq=%llu token=%llu active=%u sideband_kind=%u hidden_hash=%016llx sideband_hash=%016llx hidden_bytes=%llu sideband_bytes=%llu\n",
            (unsigned long long)packet->sequence_id,
            (unsigned long long)packet->token_index,
            packet->active_sequence_count,
            packet->sideband_kind,
            (unsigned long long)hidden_hash,
            (unsigned long long)sideband_hash,
            (unsigned long long)hidden_bytes,
            (unsigned long long)sideband_bytes);
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

static void SparkHiddenTcpCudaReleasePendingPacket(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTcpCudaPendingPacket *pending)
{
    uint32_t pending_index;
    if (state == 0 || pending == 0 || state->pending_packets == 0)
        return;
    pending_index = (uint32_t)(pending - state->pending_packets);
    if (pending_index >= state->pending_depth)
        return;
    pending->active = 0u;
    pending->next_pending_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    pending->next_free_index = state->free_pending_head;
    state->free_pending_head = pending_index;
}

static SparkStatus SparkHiddenTcpCudaFindPendingPacket(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaPendingPacket *pending;
    SparkStatus status;
    uint32_t pending_index;
    uint32_t previous_index;
    uint32_t hash_slot;
    if (state == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->pending_hash_slots == 0u || state->pending_hash_heads == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    hash_slot = SparkHiddenTcpCudaPacketHash(packet) % state->pending_hash_slots;
    previous_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    pending_index = state->pending_hash_heads[hash_slot];
    while (pending_index != SPARK_HIDDEN_TCP_CUDA_NO_PENDING)
    {
        if (pending_index >= state->pending_depth)
            return SPARK_STATUS_INTERNAL_ERROR;
        pending = &state->pending_packets[pending_index];
        if (pending->active != 0u &&
            SparkHiddenTcpCudaHeaderMatchesPacket(&pending->header,packet) != 0u)
        {
            if (previous_index == SPARK_HIDDEN_TCP_CUDA_NO_PENDING)
                state->pending_hash_heads[hash_slot] = pending->next_pending_index;
            else
                state->pending_packets[previous_index].next_pending_index =
                    pending->next_pending_index;
            status = SparkHiddenTcpCudaCopyPayloadToPacket(
                state,
                packet,
                pending->payload,
                pending->hidden_bytes,
                pending->sideband_bytes);
            SparkHiddenTcpCudaReleasePendingPacket(state,pending);
            return status;
        }
        previous_index = pending_index;
        pending_index = pending->next_pending_index;
    }
    return SPARK_STATUS_NOT_FOUND;
}

static SparkStatus SparkHiddenTcpCudaInsertPendingPacket(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTcpCudaPendingPacket *pending)
{
    uint32_t hash_slot;
    uint32_t pending_index;
    if (state == 0 || pending == 0 ||
        state->pending_packets == 0 ||
        state->pending_hash_heads == 0 ||
        state->pending_hash_slots == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    pending_index = (uint32_t)(pending - state->pending_packets);
    if (pending_index >= state->pending_depth)
        return SPARK_STATUS_INVALID_ARGUMENT;
    hash_slot = SparkHiddenTcpCudaHeaderHash(&pending->header) %
        state->pending_hash_slots;
    pending->next_pending_index = state->pending_hash_heads[hash_slot];
    state->pending_hash_heads[hash_slot] = pending_index;
    pending->active = 1u;
    return SPARK_STATUS_OK;
}

static void SparkHiddenTcpCudaInitializePendingHash(SparkHiddenTcpCudaState *state)
{
    uint32_t hash_index;
    if (state == 0 || state->pending_hash_heads == 0)
        return;
    for (hash_index = 0u; hash_index < state->pending_hash_slots; ++hash_index)
        state->pending_hash_heads[hash_index] = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
}

static void SparkHiddenTcpCudaResetPendingSlots(SparkHiddenTcpCudaState *state)
{
    uint32_t pending_index;
    if (state == 0 || state->pending_packets == 0)
        return;
    state->free_pending_head = state->pending_depth == 0u ?
        SPARK_HIDDEN_TCP_CUDA_NO_PENDING : 0u;
    for (pending_index = 0u; pending_index < state->pending_depth; ++pending_index)
    {
        state->pending_packets[pending_index].active = 0u;
        state->pending_packets[pending_index].next_pending_index =
            SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
        state->pending_packets[pending_index].next_free_index =
            pending_index + 1u < state->pending_depth ?
                pending_index + 1u : SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    }
}

static SparkStatus SparkHiddenTcpCudaReservePendingStorage(
    SparkHiddenTcpCudaState *state)
{
    uint32_t pending_depth;
    uint32_t pending_index;
    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    pending_depth = SparkHiddenTcpCudaParseUintEnv(
        "SPARKPIPE_HIDDEN_TCP_PENDING_DEPTH",
        SPARK_HIDDEN_TCP_CUDA_DEFAULT_PENDING_DEPTH);
    if (pending_depth == 0u)
        pending_depth = SPARK_HIDDEN_TCP_CUDA_DEFAULT_PENDING_DEPTH;
    state->pending_depth = pending_depth;
    state->pending_hash_slots = SparkHiddenTcpCudaParseUintEnv(
        "SPARKPIPE_HIDDEN_TCP_PENDING_HASH_SLOTS",
        SPARK_HIDDEN_TCP_CUDA_DEFAULT_PENDING_HASH_SLOTS);
    if (state->pending_hash_slots < pending_depth)
        state->pending_hash_slots = pending_depth * 2u;
    state->pending_hash_heads = (uint32_t *)malloc(
        state->pending_hash_slots * sizeof(state->pending_hash_heads[0]));
    state->pending_packets = (SparkHiddenTcpCudaPendingPacket *)calloc(
        pending_depth,
        sizeof(state->pending_packets[0]));
    if (state->pending_hash_heads == 0 || state->pending_packets == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    SparkHiddenTcpCudaInitializePendingHash(state);
    SparkHiddenTcpCudaResetPendingSlots(state);
    for (pending_index = 0u; pending_index < pending_depth; ++pending_index)
    {
        state->pending_packets[pending_index].payload =
            (uint8_t *)malloc((size_t)state->pending_payload_bytes);
        if (state->pending_packets[pending_index].payload == 0)
            return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenTcpCudaFreePendingStorage(SparkHiddenTcpCudaState *state)
{
    uint32_t pending_index;
    if (state == 0)
        return;
    if (state->pending_packets != 0)
    {
        for (pending_index = 0u; pending_index < state->pending_depth; ++pending_index)
            free(state->pending_packets[pending_index].payload);
        free(state->pending_packets);
        state->pending_packets = 0;
    }
    free(state->pending_hash_heads);
    state->pending_hash_heads = 0;
}

static SparkHiddenTcpCudaPendingPacket *SparkHiddenTcpCudaReservePendingPacket(
    SparkHiddenTcpCudaState *state)
{
    uint32_t pending_index;
    if (state == 0)
        return 0;
    pending_index = state->free_pending_head;
    if (pending_index == SPARK_HIDDEN_TCP_CUDA_NO_PENDING ||
        pending_index >= state->pending_depth)
        return 0;
    state->free_pending_head =
        state->pending_packets[pending_index].next_free_index;
    state->pending_packets[pending_index].next_free_index =
        SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    return &state->pending_packets[pending_index];
}

static void SparkHiddenTcpCudaClearIncomingFrame(SparkHiddenTcpCudaState *state)
{
    if (state == 0)
        return;
    memset(&state->incoming_header,0,sizeof(state->incoming_header));
    state->incoming_header_offset = 0u;
    state->incoming_payload_offset = 0u;
    state->incoming_payload_bytes = 0u;
    state->incoming_hidden_bytes = 0u;
    state->incoming_sideband_bytes = 0u;
    state->incoming_state = 0u;
    state->incoming_matches_request = 0u;
    state->incoming_pending_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    state->incoming_payload = 0;
}

static void SparkHiddenTcpCudaAbortIncomingFrame(SparkHiddenTcpCudaState *state)
{
    if (state == 0)
        return;
    if (state->incoming_pending_index != SPARK_HIDDEN_TCP_CUDA_NO_PENDING &&
        state->incoming_pending_index < state->pending_depth)
        SparkHiddenTcpCudaReleasePendingPacket(
            state,
            &state->pending_packets[state->incoming_pending_index]);
    SparkHiddenTcpCudaClearIncomingFrame(state);
}

static SparkStatus SparkHiddenTcpCudaStoreIncomingFramePending(
    SparkHiddenTcpCudaState *state)
{
    SparkHiddenTcpCudaPendingPacket *pending;
    SparkStatus status;
    uint64_t payload_bytes;
    uint64_t hidden_hash;
    uint64_t sideband_hash;
    if (state == 0 || state->incoming_payload == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    payload_bytes = state->incoming_hidden_bytes + state->incoming_sideband_bytes;
    pending = SparkHiddenTcpCudaReservePendingPacket(state);
    if (pending == 0)
        return SPARK_STATUS_BUSY;
    pending->header = state->incoming_header;
    pending->hidden_bytes = state->incoming_hidden_bytes;
    pending->sideband_bytes = state->incoming_sideband_bytes;
    memcpy(pending->payload,state->incoming_payload,(size_t)payload_bytes);
    status = SparkHiddenTcpCudaInsertPendingPacket(state,pending);
    if (status == SPARK_STATUS_OK && state->debug_enabled != 0u)
    {
        hidden_hash = SparkHiddenTcpCudaPayloadHash(
            pending->payload,
            pending->hidden_bytes);
        sideband_hash = SparkHiddenTcpCudaPayloadHash(
            pending->payload + pending->hidden_bytes,
            pending->sideband_bytes);
        fprintf(stderr,
            "hidden_tcp_pending_store seq=%llu token=%llu active=%u sideband_kind=%u hidden_hash=%016llx sideband_hash=%016llx hidden_bytes=%llu sideband_bytes=%llu\n",
            (unsigned long long)pending->header.sequence_id,
            (unsigned long long)pending->header.token_index,
            pending->header.active_sequence_count,
            pending->header.sideband_kind,
            (unsigned long long)hidden_hash,
            (unsigned long long)sideband_hash,
            (unsigned long long)pending->hidden_bytes,
            (unsigned long long)pending->sideband_bytes);
    }
    if (status != SPARK_STATUS_OK)
        SparkHiddenTcpCudaReleasePendingPacket(state,pending);
    SparkHiddenTcpCudaClearIncomingFrame(state);
    return status;
}

static SparkStatus SparkHiddenTcpCudaBeginIncomingPayload(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaPendingPacket *pending;
    SparkStatus status;
    if (state == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTcpCudaValidateIncomingHeader(
        state,
        &state->incoming_header,
        &state->incoming_hidden_bytes,
        &state->incoming_sideband_bytes);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenTcpCudaCloseSocket(state);
        SparkHiddenTcpCudaAbortIncomingFrame(state);
        return SPARK_STATUS_BUSY;
    }
    state->incoming_payload_bytes =
        state->incoming_hidden_bytes + state->incoming_sideband_bytes;
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_tcp_recv_header seq=%llu token=%llu active=%u sideband_kind=%u sideband_bps=%u hidden_bytes=%llu sideband_bytes=%llu match=%u\n",
            (unsigned long long)state->incoming_header.sequence_id,
            (unsigned long long)state->incoming_header.token_index,
            state->incoming_header.active_sequence_count,
            state->incoming_header.sideband_kind,
            state->incoming_header.sideband_bytes_per_sequence,
            (unsigned long long)state->incoming_hidden_bytes,
            (unsigned long long)state->incoming_sideband_bytes,
            SparkHiddenTcpCudaHeaderMatchesPacket(
                &state->incoming_header,
                packet));
    }
    if (SparkHiddenTcpCudaHeaderMatchesPacket(&state->incoming_header,packet) != 0u)
    {
        state->incoming_matches_request = 1u;
        state->incoming_payload = state->host_buffer;
    }
    else
    {
        pending = SparkHiddenTcpCudaReservePendingPacket(state);
        if (pending == 0)
            return SPARK_STATUS_BUSY;
        state->incoming_pending_index =
            (uint32_t)(pending - state->pending_packets);
        state->incoming_payload = pending->payload;
    }
    state->incoming_state = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaProgressIncomingFrame(
    SparkHiddenTcpCudaState *state,
    SparkHiddenTransportPacket *packet,
    uint32_t *delivered)
{
    SparkHiddenTcpCudaPendingPacket *pending;
    SparkStatus status;
    uint64_t transferred;
    uint64_t remaining;
    if (state == 0 || packet == 0 || delivered == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    *delivered = 0u;
    while (state->incoming_header_offset < sizeof(state->incoming_header))
    {
        remaining = sizeof(state->incoming_header) -
            state->incoming_header_offset;
        status = SparkHiddenTcpCudaReadSome(
            state->socket_fd,
            ((uint8_t *)&state->incoming_header) + state->incoming_header_offset,
            remaining,
            &transferred);
        if (status == SPARK_STATUS_BUSY)
            return SPARK_STATUS_BUSY;
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenTcpCudaCloseSocket(state);
            SparkHiddenTcpCudaAbortIncomingFrame(state);
            return SPARK_STATUS_BUSY;
        }
        state->incoming_header_offset += transferred;
    }
    if (state->incoming_state == 0u)
    {
        status = SparkHiddenTcpCudaBeginIncomingPayload(state,packet);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    while (state->incoming_payload_offset < state->incoming_payload_bytes)
    {
        remaining = state->incoming_payload_bytes - state->incoming_payload_offset;
        status = SparkHiddenTcpCudaReadSome(
            state->socket_fd,
            state->incoming_payload + state->incoming_payload_offset,
            remaining,
            &transferred);
        if (status == SPARK_STATUS_BUSY)
            return SPARK_STATUS_BUSY;
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenTcpCudaCloseSocket(state);
            SparkHiddenTcpCudaAbortIncomingFrame(state);
            return SPARK_STATUS_BUSY;
        }
        state->incoming_payload_offset += transferred;
    }
    if (state->incoming_matches_request != 0u)
    {
        if (SparkHiddenTcpCudaHeaderMatchesPacket(
                &state->incoming_header,
                packet) != 0u)
        {
            status = SparkHiddenTcpCudaCopyPayloadToPacket(
                state,
                packet,
                state->incoming_payload,
                state->incoming_hidden_bytes,
                state->incoming_sideband_bytes);
            SparkHiddenTcpCudaClearIncomingFrame(state);
            if (status == SPARK_STATUS_OK)
                *delivered = 1u;
            return status;
        }
        status = SparkHiddenTcpCudaStoreIncomingFramePending(state);
        return status;
    }
    if (state->incoming_pending_index >= state->pending_depth)
    {
        SparkHiddenTcpCudaAbortIncomingFrame(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    pending = &state->pending_packets[state->incoming_pending_index];
    pending->header = state->incoming_header;
    pending->hidden_bytes = state->incoming_hidden_bytes;
    pending->sideband_bytes = state->incoming_sideband_bytes;
    state->incoming_pending_index = SPARK_HIDDEN_TCP_CUDA_NO_PENDING;
    status = SparkHiddenTcpCudaInsertPendingPacket(state,pending);
    SparkHiddenTcpCudaClearIncomingFrame(state);
    return status;
}

static SparkStatus SparkHiddenTcpCudaEnsureSocket(SparkHiddenTcpCudaState *state)
{
    const char *host;
    int fd;
    if (state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->socket_fd >= 0)
        return SparkHiddenTcpCudaFinishConnect(state);
    if (state->is_sender != 0u)
    {
        host = getenv("SPARKPIPE_PP13_TRANSPORT_HOST_OVERRIDE");
        if (host == 0 || host[0] == '\0')
            host = state->sink_host;
        return SparkHiddenTcpCudaBeginConnect(
            state,
            host,
            state->port_base + (uint32_t)state->sink_rank);
    }
    fd = accept(state->listen_fd,0,0);
    if (fd < 0 && (errno == EINTR ||
            errno == EAGAIN ||
            errno == EWOULDBLOCK))
        return SPARK_STATUS_BUSY;
    state->socket_fd = fd;
    state->socket_connecting = 0u;
    if (state->socket_fd >= 0)
    {
        (void)SparkHiddenTcpCudaSetNonblocking(state->socket_fd);
        SparkHiddenTcpCudaConfigureSocket(state->socket_fd);
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
    state->pending_payload_bytes = state->host_buffer_bytes -
        sizeof(SparkHiddenTcpCudaHeader);
    state->host_buffer = (uint8_t *)malloc((size_t)state->host_buffer_bytes);
    state->debug_enabled = SparkHiddenTcpCudaDebugEnabled();
    if (state->host_buffer == 0)
    {
        if (state->listen_fd >= 0)
            close(state->listen_fd);
        free(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (SparkHiddenTcpCudaReservePendingStorage(state) != SPARK_STATUS_OK)
    {
        SparkHiddenTcpCudaFreePendingStorage(state);
        free(state->host_buffer);
        if (state->listen_fd >= 0)
            close(state->listen_fd);
        free(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkHiddenTcpCudaClearIncomingFrame(state);
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
    SparkHiddenTcpCudaFreePendingStorage(state);
    free(state->host_buffer);
    free(state);
}

static SparkStatus SparkHiddenTcpCudaSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaState *state;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;
    uint64_t hidden_hash;
    uint64_t sideband_hash;
    uint64_t offset;
    uint64_t transferred;
    uint64_t remaining;
    SparkStatus status;
    state = (SparkHiddenTcpCudaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkHiddenTransportValidatePacket(&state->endpoint,packet);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkHiddenTcpCudaEnsureSocket(state);
    if (status == SPARK_STATUS_ROUTE_NOT_FOUND)
        return SPARK_STATUS_BUSY;
    if (status == SPARK_STATUS_BUSY)
        return SPARK_STATUS_BUSY;
    if (status != SPARK_STATUS_OK)
        return status;
    if (state->outgoing_active == 0u)
    {
        hidden_bytes = (uint64_t)packet->bytes_per_sequence *
            (uint64_t)packet->active_sequence_count;
        sideband_bytes = (uint64_t)packet->sideband_bytes_per_sequence *
            (uint64_t)packet->active_sequence_count;
        if (sizeof(state->outgoing_header) + hidden_bytes + sideband_bytes >
            state->host_buffer_bytes)
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
            return SPARK_STATUS_IO_ERROR;
        memset(&state->outgoing_header,0,sizeof(state->outgoing_header));
        state->outgoing_header.magic = SPARK_HIDDEN_TCP_CUDA_MAGIC;
        state->outgoing_header.active_sequence_count =
            packet->active_sequence_count;
        state->outgoing_header.hidden_dimension = packet->hidden_dimension;
        state->outgoing_header.bytes_per_sequence = packet->bytes_per_sequence;
        state->outgoing_header.sequence_id = packet->sequence_id;
        state->outgoing_header.token_index = packet->token_index;
        state->outgoing_header.sideband_kind = packet->sideband_kind;
        state->outgoing_header.sideband_bytes_per_sequence =
            packet->sideband_bytes_per_sequence;
        memcpy(state->host_buffer,&state->outgoing_header,
            sizeof(state->outgoing_header));
        offset = sizeof(state->outgoing_header);
        if (cudaMemcpy(state->host_buffer + offset,packet->hidden_bf16,
                (size_t)hidden_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
            return SPARK_STATUS_IO_ERROR;
        offset += hidden_bytes;
        if (sideband_bytes != 0u &&
            cudaMemcpy(state->host_buffer + offset,packet->sideband_payload,
                (size_t)sideband_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
            return SPARK_STATUS_IO_ERROR;
        state->outgoing_bytes =
            sizeof(state->outgoing_header) + hidden_bytes + sideband_bytes;
        state->outgoing_offset = 0u;
        state->outgoing_active = 1u;
        if (state->debug_enabled != 0u)
        {
            hidden_hash = SparkHiddenTcpCudaPayloadHash(
                state->host_buffer + sizeof(state->outgoing_header),
                hidden_bytes);
            sideband_hash = SparkHiddenTcpCudaPayloadHash(
                state->host_buffer + sizeof(state->outgoing_header) + hidden_bytes,
                sideband_bytes);
            fprintf(stderr,
                "hidden_tcp_send_header seq=%llu token=%llu active=%u sideband_kind=%u sideband_bps=%u hidden_hash=%016llx sideband_hash=%016llx hidden_bytes=%llu sideband_bytes=%llu total=%llu\n",
                (unsigned long long)state->outgoing_header.sequence_id,
                (unsigned long long)state->outgoing_header.token_index,
                state->outgoing_header.active_sequence_count,
                state->outgoing_header.sideband_kind,
                state->outgoing_header.sideband_bytes_per_sequence,
                (unsigned long long)hidden_hash,
                (unsigned long long)sideband_hash,
                (unsigned long long)hidden_bytes,
                (unsigned long long)sideband_bytes,
                (unsigned long long)state->outgoing_bytes);
        }
    }
    else if (SparkHiddenTcpCudaHeaderMatchesPacket(
            &state->outgoing_header,
            packet) == 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    while (state->outgoing_offset < state->outgoing_bytes)
    {
        remaining = state->outgoing_bytes - state->outgoing_offset;
        status = SparkHiddenTcpCudaWriteSome(
            state->socket_fd,
            state->host_buffer + state->outgoing_offset,
            remaining,
            &transferred);
        if (status == SPARK_STATUS_BUSY)
            return SPARK_STATUS_BUSY;
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenTcpCudaCloseSocket(state);
            state->outgoing_active = 0u;
            state->outgoing_offset = 0u;
            state->outgoing_bytes = 0u;
            return SPARK_STATUS_BUSY;
        }
        state->outgoing_offset += transferred;
    }
    memset(&state->completion,0,sizeof(state->completion));
    state->completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    state->completion.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    state->completion.status = SPARK_STATUS_OK;
    state->completion.active_sequence_count = packet->active_sequence_count;
    state->completion.sequence_id = packet->sequence_id;
    state->completion.token_index = packet->token_index;
    state->completion.transfer_bytes =
        state->outgoing_bytes - sizeof(state->outgoing_header);
    state->completion_ready = 1u;
    state->outgoing_active = 0u;
    state->outgoing_offset = 0u;
    state->outgoing_bytes = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenTcpCudaState *state;
    uint32_t delivered;
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
    status = SparkHiddenTcpCudaFindPendingPacket(state,packet);
    if (status == SPARK_STATUS_OK)
        return SPARK_STATUS_OK;
    if (status != SPARK_STATUS_NOT_FOUND)
        return status;
    for (;;)
    {
        status = SparkHiddenTcpCudaProgressIncomingFrame(
            state,
            packet,
            &delivered);
        if (status != SPARK_STATUS_OK)
            return status;
        if (delivered != 0u)
            return SPARK_STATUS_OK;
        status = SparkHiddenTcpCudaFindPendingPacket(state,packet);
        if (status == SPARK_STATUS_OK)
            return SPARK_STATUS_OK;
        if (status != SPARK_STATUS_NOT_FOUND)
            return status;
    }
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

static SparkStatus SparkHiddenTcpCudaAppendPollDescriptor(
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count,
    int fd,
    uint32_t events)
{
    SparkHiddenTransportPollDescriptor *descriptor;
    if (descriptor_count == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (fd < 0 || events == 0u)
        return SPARK_STATUS_OK;
    if (*descriptor_count >= descriptor_capacity || descriptors == 0)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    descriptor = &descriptors[*descriptor_count];
    memset(descriptor,0,sizeof(*descriptor));
    descriptor->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    descriptor->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES;
    descriptor->fd = fd;
    descriptor->events = events;
    *descriptor_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTcpCudaGetPollDescriptors(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out)
{
    SparkHiddenTcpCudaState *state;
    SparkStatus status;
    uint32_t descriptor_count;
    if (transport_state == 0 || descriptor_count_out == 0 ||
        (descriptors == 0 && descriptor_capacity != 0u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    state = (SparkHiddenTcpCudaState *)transport_state;
    descriptor_count = 0u;
    if (state->is_sender != 0u &&
        (state->outgoing_active != 0u || state->socket_connecting != 0u))
    {
        status = SparkHiddenTcpCudaAppendPollDescriptor(
            descriptors,
            descriptor_capacity,
            &descriptor_count,
            state->socket_fd,
            SPARK_HIDDEN_TRANSPORT_POLL_WRITE);
    }
    else if (state->is_sender != 0u)
        status = SPARK_STATUS_OK;
    else if (state->socket_fd >= 0)
    {
        status = SparkHiddenTcpCudaAppendPollDescriptor(
            descriptors,
            descriptor_capacity,
            &descriptor_count,
            state->socket_fd,
            SPARK_HIDDEN_TRANSPORT_POLL_READ);
    }
    else
    {
        status = SparkHiddenTcpCudaAppendPollDescriptor(
            descriptors,
            descriptor_capacity,
            &descriptor_count,
            state->listen_fd,
            SPARK_HIDDEN_TRANSPORT_POLL_READ);
    }
    if (status != SPARK_STATUS_OK)
        return status;
    *descriptor_count_out = descriptor_count;
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
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PIPELINE_HOST_STAGED_CAPS |
        SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS;
    transport_interface.initialize = SparkHiddenTcpCudaInitialize;
    transport_interface.destroy = SparkHiddenTcpCudaDestroy;
    transport_interface.post_receive = SparkHiddenTcpCudaPostReceive;
    transport_interface.send = SparkHiddenTcpCudaSend;
    transport_interface.poll = SparkHiddenTcpCudaPoll;
    transport_interface.post_receive_batch = SparkHiddenTcpCudaPostReceiveBatch;
    transport_interface.send_batch = SparkHiddenTcpCudaSendBatch;
    transport_interface.get_poll_descriptors = SparkHiddenTcpCudaGetPollDescriptors;
    return &transport_interface;
}

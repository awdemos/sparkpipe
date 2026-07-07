#include "sparkpipe/spark_hidden_transport.h"

#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SPARK_HIDDEN_GDR_CONTROL_MAGIC 0x53475055u
#define SPARK_HIDDEN_GDR_CONTROL_VERSION 1u
#define SPARK_HIDDEN_GDR_DEFAULT_CONTROL_PORT_BASE 55700u
#define SPARK_HIDDEN_GDR_DEFAULT_LANE_COUNT 8u
#define SPARK_HIDDEN_GDR_MAX_LANE_COUNT 32u
#define SPARK_HIDDEN_GDR_MAX_PENDING_RECEIVE_COUNT 64u
#define SPARK_HIDDEN_GDR_MAX_REMOTE_RECEIVE_COUNT 64u
#define SPARK_HIDDEN_GDR_MR_CACHE_COUNT 64u
#define SPARK_HIDDEN_GDR_CONNECT_RETRY_MS 50u
#define SPARK_HIDDEN_GDR_CONTROL_HOST_BYTES 64u
#define SPARK_HIDDEN_GDR_POLL_TIMEOUT_MS 1000u

#define SPARK_HIDDEN_GDR_CONTROL_HELLO 1u
#define SPARK_HIDDEN_GDR_CONTROL_RECEIVE_READY 2u
#define SPARK_HIDDEN_GDR_CONTROL_TRANSFER_COMPLETE 3u

#define SPARK_HIDDEN_GDR_WR_ID_REGION_HIDDEN 1ull
#define SPARK_HIDDEN_GDR_WR_ID_REGION_SIDEBAND 2ull

#define SPARK_HIDDEN_GDR_NO_INDEX 0xffffffffu

typedef struct SparkHiddenGdrMemoryRegionDescriptor
{
    uint64_t address;
    uint64_t bytes;
    uint32_t rkey;
    uint32_t reserved;
} SparkHiddenGdrMemoryRegionDescriptor;

typedef struct SparkHiddenGdrControlMessage
{
    uint32_t magic;
    uint32_t version;
    uint32_t type;
    uint32_t status;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t active_sequence_count;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
    uint32_t reserved;
    SparkHiddenGdrMemoryRegionDescriptor hidden;
    SparkHiddenGdrMemoryRegionDescriptor sideband;
} SparkHiddenGdrControlMessage;

typedef struct SparkHiddenGdrQueuePairWireInfo
{
    uint32_t qp_number;
    uint32_t packet_sequence_number;
    uint16_t lid;
    uint16_t reserved;
    uint8_t gid[16u];
} SparkHiddenGdrQueuePairWireInfo;

typedef struct SparkHiddenGdrLane
{
    struct ibv_cq *completion_queue;
    struct ibv_qp *queue_pair;
    SparkHiddenGdrQueuePairWireInfo local_info;
    SparkHiddenGdrQueuePairWireInfo remote_info;
} SparkHiddenGdrLane;

typedef struct SparkHiddenGdrCachedMemoryRegion
{
    const void *pointer;
    uint64_t bytes;
    uint64_t last_use_epoch;
    struct ibv_mr *memory_region;
} SparkHiddenGdrCachedMemoryRegion;

typedef struct SparkHiddenGdrPendingReceive
{
    uint32_t active;
    uint32_t complete;
    uint32_t advertised;
    SparkHiddenTransportPacket packet_snapshot;
    SparkHiddenGdrMemoryRegionDescriptor hidden_descriptor;
    SparkHiddenGdrMemoryRegionDescriptor sideband_descriptor;
    struct ibv_mr *hidden_memory_region;
    struct ibv_mr *sideband_memory_region;
} SparkHiddenGdrPendingReceive;

typedef struct SparkHiddenGdrRemoteReceive
{
    uint32_t active;
    uint32_t used;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t active_sequence_count;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
    SparkHiddenGdrMemoryRegionDescriptor hidden_descriptor;
    SparkHiddenGdrMemoryRegionDescriptor sideband_descriptor;
} SparkHiddenGdrRemoteReceive;

typedef struct SparkHiddenGdrState
{
    SparkHiddenTransportEndpoint endpoint;
    int32_t local_rank;
    int32_t source_rank;
    int32_t sink_rank;
    uint32_t is_sender;
    uint32_t lane_count;
    uint32_t control_port_base;
    uint8_t verbs_port;
    int32_t gid_index;
    int listen_fd;
    int control_fd;
    int event_fd;
    uint32_t debug_enabled;
    char source_host[SPARK_HIDDEN_GDR_CONTROL_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_GDR_CONTROL_HOST_BYTES];
    struct ibv_context *verbs_context;
    struct ibv_pd *protection_domain;
    union ibv_gid local_gid;
    uint16_t local_lid;
    SparkHiddenGdrLane lanes[SPARK_HIDDEN_GDR_MAX_LANE_COUNT];
    SparkHiddenGdrCachedMemoryRegion cached_regions[SPARK_HIDDEN_GDR_MR_CACHE_COUNT];
    uint64_t memory_region_epoch;
    SparkHiddenGdrPendingReceive pending_receives[SPARK_HIDDEN_GDR_MAX_PENDING_RECEIVE_COUNT];
    SparkHiddenGdrRemoteReceive remote_receives[SPARK_HIDDEN_GDR_MAX_REMOTE_RECEIVE_COUNT];
    SparkHiddenTransportCompletion completion;
    uint32_t completion_ready;
} SparkHiddenGdrState;

static uint32_t SparkHiddenGdrParseUintEnv(
    const char *name,
    uint32_t fallback)
{
    const char *text;
    char *end;
    unsigned long value;

    text = getenv(name);
    if (text == 0 || text[0] == '\0')
    {
        return fallback;
    }
    end = 0;
    value = strtoul(text, &end, 10);
    if (end == 0 || *end != '\0' || value > 0xfffffffful)
    {
        return fallback;
    }
    return (uint32_t)value;
}

static int32_t SparkHiddenGdrParseIntEnv(
    const char *name,
    int32_t fallback)
{
    const char *text;
    char *end;
    long value;

    text = getenv(name);
    if (text == 0 || text[0] == '\0')
    {
        return fallback;
    }
    end = 0;
    value = strtol(text, &end, 10);
    if (end == 0 || *end != '\0' || value < -2147483647l ||
        value > 2147483647l)
    {
        return fallback;
    }
    return (int32_t)value;
}

static int SparkHiddenGdrCloseFd(int fd)
{
    if (fd >= 0)
    {
        (void)close(fd);
    }
    return -1;
}

static void SparkHiddenGdrSignalEvent(SparkHiddenGdrState *state)
{
    ssize_t written;
    uint64_t value;

    if (state == 0 || state->event_fd < 0)
    {
        return;
    }
    value = 1u;
    written = write(state->event_fd, &value, sizeof(value));
    if (written != (ssize_t)sizeof(value))
    {
        return;
    }
}

static void SparkHiddenGdrDrainEvent(SparkHiddenGdrState *state)
{
    uint64_t value;

    if (state == 0 || state->event_fd < 0)
    {
        return;
    }
    while (read(state->event_fd, &value, sizeof(value)) == sizeof(value))
    {
    }
}

static SparkStatus SparkHiddenGdrSetNonblocking(int fd)
{
    int flags;

    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenGdrConfigureSocket(int fd)
{
    int value;

    if (fd < 0)
    {
        return;
    }
    value = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value));
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &value, sizeof(value));
}

static int SparkHiddenGdrListen(uint32_t port)
{
    struct sockaddr_in address;
    int fd;
    int value;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        return -1;
    }
    value = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(fd, 1) != 0)
    {
        (void)close(fd);
        return -1;
    }
    return fd;
}

static SparkStatus SparkHiddenGdrReadFull(
    int fd,
    void *buffer,
    uint64_t bytes)
{
    uint8_t *cursor;
    uint64_t done;
    ssize_t result;

    cursor = (uint8_t *)buffer;
    done = 0u;
    while (done < bytes)
    {
        result = read(fd, cursor + done, (size_t)(bytes - done));
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        if (result <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        done += (uint64_t)result;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrWriteFull(
    int fd,
    const void *buffer,
    uint64_t bytes)
{
    const uint8_t *cursor;
    uint64_t done;
    ssize_t result;

    cursor = (const uint8_t *)buffer;
    done = 0u;
    while (done < bytes)
    {
        result = write(fd, cursor + done, (size_t)(bytes - done));
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        if (result <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        done += (uint64_t)result;
    }
    return SPARK_STATUS_OK;
}

static int32_t SparkHiddenGdrRankFromHost(const char *host)
{
    uint32_t tail;
    char extra;

    if (host == 0 || host[0] != 's' || host[1] != 'p' ||
        host[2] != 'a' || host[3] != 'r' || host[4] != 'k' ||
        host[5] == '\0')
    {
        extra = '\0';
        if (sscanf(host, "10.10.100.%u%c", &tail, &extra) == 1 &&
            tail >= 10u && tail <= 22u)
        {
            return (int32_t)(tail - 10u);
        }
        return -1;
    }
    if (host[5] >= '0' && host[5] <= '9' && host[6] == '\0')
    {
        return (int32_t)(host[5] - '0');
    }
    if (host[5] >= 'a' && host[5] <= 'c' && host[6] == '\0')
    {
        return (int32_t)(10 + (host[5] - 'a'));
    }
    return -1;
}

static SparkStatus SparkHiddenGdrParseRoute(
    const char *route_name,
    char *source_host,
    char *sink_host)
{
    const char *middle;
    const char *suffix;
    uint64_t source_bytes;
    uint64_t sink_bytes;

    if (route_name == 0 || source_host == 0 || sink_host == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    middle = strstr(route_name, "_to_");
    suffix = strstr(route_name, "_hidden");
    if (middle == 0 || suffix == 0 || middle >= suffix)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    source_bytes = (uint64_t)(middle - route_name);
    sink_bytes = (uint64_t)(suffix - (middle + 4));
    if (source_bytes == 0u || sink_bytes == 0u ||
        source_bytes >= SPARK_HIDDEN_GDR_CONTROL_HOST_BYTES ||
        sink_bytes >= SPARK_HIDDEN_GDR_CONTROL_HOST_BYTES)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(source_host, route_name, (size_t)source_bytes);
    source_host[source_bytes] = '\0';
    memcpy(sink_host, middle + 4, (size_t)sink_bytes);
    sink_host[sink_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrConnectControl(
    SparkHiddenGdrState *state)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16u];
    int fd;
    const char *host;

    if (state->is_sender == 0u)
    {
        state->listen_fd = SparkHiddenGdrListen(
            state->control_port_base + (uint32_t)state->sink_rank);
        if (state->listen_fd < 0)
        {
            return SPARK_STATUS_ROUTE_NOT_FOUND;
        }
        fd = accept(state->listen_fd, 0, 0);
        if (fd < 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        state->control_fd = fd;
        SparkHiddenGdrConfigureSocket(state->control_fd);
        return SPARK_STATUS_OK;
    }

    host = state->sink_host;
    while (state->control_fd < 0)
    {
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        snprintf(port_text, sizeof(port_text), "%u",
            state->control_port_base + (uint32_t)state->sink_rank);
        result = 0;
        if (getaddrinfo(host, port_text, &hints, &result) != 0 || result == 0)
        {
            (void)poll(0, 0, SPARK_HIDDEN_GDR_CONNECT_RETRY_MS);
            continue;
        }
        for (entry = result; entry != 0 && state->control_fd < 0;
             entry = entry->ai_next)
        {
            fd = socket(entry->ai_family, entry->ai_socktype,
                entry->ai_protocol);
            if (fd < 0)
            {
                continue;
            }
            if (connect(fd, entry->ai_addr, entry->ai_addrlen) == 0)
            {
                state->control_fd = fd;
                SparkHiddenGdrConfigureSocket(state->control_fd);
            }
            else
            {
                (void)close(fd);
            }
        }
        freeaddrinfo(result);
        if (state->control_fd < 0)
        {
            (void)poll(0, 0, SPARK_HIDDEN_GDR_CONNECT_RETRY_MS);
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrOpenVerbsDevice(SparkHiddenGdrState *state)
{
    struct ibv_device **devices;
    struct ibv_device *selected_device;
    const char *requested_name;
    int count;
    int index;
    struct ibv_port_attr port_attributes;

    requested_name = getenv("SPARKPIPE_HIDDEN_GDR_IB_DEVICE");
    devices = ibv_get_device_list(&count);
    if (devices == 0 || count <= 0)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    selected_device = 0;
    for (index = 0; index < count; ++index)
    {
        if (requested_name == 0 || requested_name[0] == '\0' ||
            strcmp(ibv_get_device_name(devices[index]), requested_name) == 0)
        {
            selected_device = devices[index];
            break;
        }
    }
    if (selected_device != 0)
    {
        state->verbs_context = ibv_open_device(selected_device);
    }
    ibv_free_device_list(devices);
    if (state->verbs_context == 0)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->protection_domain = ibv_alloc_pd(state->verbs_context);
    if (state->protection_domain == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    if (ibv_query_port(state->verbs_context, state->verbs_port,
            &port_attributes) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    state->local_lid = port_attributes.lid;
    memset(&state->local_gid, 0, sizeof(state->local_gid));
    if (ibv_query_gid(state->verbs_context, state->verbs_port,
            state->gid_index, &state->local_gid) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrModifyQueuePairToInit(
    SparkHiddenGdrState *state,
    SparkHiddenGdrLane *lane)
{
    struct ibv_qp_attr attributes;

    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.pkey_index = 0;
    attributes.port_num = state->verbs_port;
    attributes.qp_access_flags = IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(lane->queue_pair, &attributes,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                IBV_QP_ACCESS_FLAGS) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrCreateQueuePairs(SparkHiddenGdrState *state)
{
    struct ibv_qp_init_attr init_attributes;
    uint32_t lane_index;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        SparkHiddenGdrLane *lane;

        lane = &state->lanes[lane_index];
        lane->completion_queue = ibv_create_cq(state->verbs_context, 128, 0,
            0, 0);
        if (lane->completion_queue == 0)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        memset(&init_attributes, 0, sizeof(init_attributes));
        init_attributes.send_cq = lane->completion_queue;
        init_attributes.recv_cq = lane->completion_queue;
        init_attributes.qp_type = IBV_QPT_RC;
        init_attributes.cap.max_send_wr = 128;
        init_attributes.cap.max_recv_wr = 1;
        init_attributes.cap.max_send_sge = 1;
        init_attributes.cap.max_recv_sge = 1;
        lane->queue_pair = ibv_create_qp(state->protection_domain,
            &init_attributes);
        if (lane->queue_pair == 0)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        if (SparkHiddenGdrModifyQueuePairToInit(state, lane) !=
            SPARK_STATUS_OK)
        {
            return SPARK_STATUS_DRIVER_LOAD_ERROR;
        }
        memset(&lane->local_info, 0, sizeof(lane->local_info));
        lane->local_info.qp_number = lane->queue_pair->qp_num;
        lane->local_info.packet_sequence_number =
            0x778800u + ((uint32_t)getpid() & 0xfffu) + lane_index;
        lane->local_info.lid = state->local_lid;
        memcpy(lane->local_info.gid, state->local_gid.raw,
            sizeof(lane->local_info.gid));
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrExchangeQueuePairInfo(
    SparkHiddenGdrState *state)
{
    SparkHiddenGdrQueuePairWireInfo local_infos[SPARK_HIDDEN_GDR_MAX_LANE_COUNT];
    SparkHiddenGdrQueuePairWireInfo remote_infos[SPARK_HIDDEN_GDR_MAX_LANE_COUNT];
    uint32_t bytes;
    uint32_t lane_index;
    SparkStatus status;

    memset(local_infos, 0, sizeof(local_infos));
    memset(remote_infos, 0, sizeof(remote_infos));
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        local_infos[lane_index] = state->lanes[lane_index].local_info;
    }
    bytes = state->lane_count *
        (uint32_t)sizeof(SparkHiddenGdrQueuePairWireInfo);
    if (state->is_sender != 0u)
    {
        status = SparkHiddenGdrWriteFull(state->control_fd,
            &state->lane_count, sizeof(state->lane_count));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenGdrWriteFull(state->control_fd,
            local_infos, bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenGdrReadFull(state->control_fd,
            remote_infos, bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    else
    {
        uint32_t remote_lane_count;

        remote_lane_count = 0u;
        status = SparkHiddenGdrReadFull(state->control_fd,
            &remote_lane_count, sizeof(remote_lane_count));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (remote_lane_count != state->lane_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkHiddenGdrReadFull(state->control_fd,
            remote_infos, bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenGdrWriteFull(state->control_fd,
            local_infos, bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        state->lanes[lane_index].remote_info = remote_infos[lane_index];
    }
    return SparkHiddenGdrSetNonblocking(state->control_fd);
}

static SparkStatus SparkHiddenGdrModifyQueuePairToReady(
    SparkHiddenGdrState *state,
    SparkHiddenGdrLane *lane)
{
    struct ibv_qp_attr attributes;
    union ibv_gid remote_gid;

    memcpy(remote_gid.raw, lane->remote_info.gid, sizeof(remote_gid.raw));
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = IBV_MTU_4096;
    attributes.dest_qp_num = lane->remote_info.qp_number;
    attributes.rq_psn = lane->remote_info.packet_sequence_number;
    attributes.max_dest_rd_atomic = 1;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.grh.dgid = remote_gid;
    attributes.ah_attr.grh.sgid_index = state->gid_index;
    attributes.ah_attr.grh.hop_limit = 1;
    attributes.ah_attr.dlid = lane->remote_info.lid;
    attributes.ah_attr.sl = 0;
    attributes.ah_attr.src_path_bits = 0;
    attributes.ah_attr.port_num = state->verbs_port;
    if (ibv_modify_qp(lane->queue_pair, &attributes,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }

    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.sq_psn = lane->local_info.packet_sequence_number;
    attributes.max_rd_atomic = 1;
    if (ibv_modify_qp(lane->queue_pair, &attributes,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                IBV_QP_MAX_QP_RD_ATOMIC) != 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrReadyQueuePairs(SparkHiddenGdrState *state)
{
    uint32_t lane_index;
    SparkStatus status;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        status = SparkHiddenGdrModifyQueuePairToReady(state,
            &state->lanes[lane_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenGdrDeregisterCachedMemoryRegions(
    SparkHiddenGdrState *state)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_GDR_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region != 0)
        {
            ibv_dereg_mr(state->cached_regions[index].memory_region);
        }
        memset(&state->cached_regions[index], 0,
            sizeof(state->cached_regions[index]));
    }
}

static SparkStatus SparkHiddenGdrGetCachedMemoryRegion(
    SparkHiddenGdrState *state,
    const void *pointer,
    uint64_t bytes,
    struct ibv_mr **memory_region_out)
{
    uint32_t index;
    uint32_t best_index;
    uint64_t best_epoch;
    int access_flags;
    struct ibv_mr *memory_region;

    if (state == 0 || pointer == 0 || bytes == 0u ||
        memory_region_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->memory_region_epoch += 1u;
    for (index = 0u; index < SPARK_HIDDEN_GDR_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region != 0 &&
            state->cached_regions[index].pointer == pointer &&
            state->cached_regions[index].bytes == bytes)
        {
            state->cached_regions[index].last_use_epoch =
                state->memory_region_epoch;
            *memory_region_out = state->cached_regions[index].memory_region;
            return SPARK_STATUS_OK;
        }
    }

    best_index = 0u;
    best_epoch = UINT64_MAX;
    for (index = 0u; index < SPARK_HIDDEN_GDR_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region == 0)
        {
            best_index = index;
            best_epoch = 0u;
            break;
        }
        if (state->cached_regions[index].last_use_epoch < best_epoch)
        {
            best_epoch = state->cached_regions[index].last_use_epoch;
            best_index = index;
        }
    }
    if (state->cached_regions[best_index].memory_region != 0)
    {
        ibv_dereg_mr(state->cached_regions[best_index].memory_region);
        memset(&state->cached_regions[best_index], 0,
            sizeof(state->cached_regions[best_index]));
    }
    access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    errno = 0;
    memory_region = ibv_reg_mr(state->protection_domain, (void *)pointer,
        (size_t)bytes, access_flags);
    if (memory_region == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    state->cached_regions[best_index].pointer = pointer;
    state->cached_regions[best_index].bytes = bytes;
    state->cached_regions[best_index].last_use_epoch =
        state->memory_region_epoch;
    state->cached_regions[best_index].memory_region = memory_region;
    *memory_region_out = memory_region;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrRegisterReceiveRegion(
    SparkHiddenGdrState *state,
    const void *pointer,
    uint64_t bytes,
    SparkHiddenGdrMemoryRegionDescriptor *descriptor,
    struct ibv_mr **memory_region_out)
{
    int access_flags;
    struct ibv_mr *memory_region;

    if (descriptor == 0 || memory_region_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(descriptor, 0, sizeof(*descriptor));
    *memory_region_out = 0;
    if (bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (state == 0 || pointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    memory_region = ibv_reg_mr(state->protection_domain, (void *)pointer,
        (size_t)bytes, access_flags);
    if (memory_region == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    descriptor->address = (uint64_t)(uintptr_t)pointer;
    descriptor->bytes = bytes;
    descriptor->rkey = memory_region->rkey;
    *memory_region_out = memory_region;
    return SPARK_STATUS_OK;
}

static uint64_t SparkHiddenGdrPacketHiddenBytes(
    const SparkHiddenTransportPacket *packet)
{
    return (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
}

static uint64_t SparkHiddenGdrPacketSidebandBytes(
    const SparkHiddenTransportPacket *packet)
{
    if ((packet->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD) == 0u)
    {
        return 0u;
    }
    return (uint64_t)packet->sideband_bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
}

static uint32_t SparkHiddenGdrRemoteReceiveMatchesPacket(
    const SparkHiddenGdrRemoteReceive *receive,
    const SparkHiddenTransportPacket *packet)
{
    return receive != 0 && packet != 0 && receive->active != 0u &&
        receive->used == 0u && receive->sequence_id == packet->sequence_id &&
        receive->token_index == packet->token_index &&
        receive->active_sequence_count == packet->active_sequence_count &&
        receive->sideband_kind == packet->sideband_kind &&
        receive->sideband_bytes_per_sequence == packet->sideband_bytes_per_sequence;
}

static uint32_t SparkHiddenGdrPendingReceiveMatchesPacket(
    const SparkHiddenGdrPendingReceive *receive,
    const SparkHiddenTransportPacket *packet)
{
    return receive != 0 && packet != 0 && receive->active != 0u &&
        receive->packet_snapshot.sequence_id == packet->sequence_id &&
        receive->packet_snapshot.token_index == packet->token_index &&
        receive->packet_snapshot.active_sequence_count ==
            packet->active_sequence_count &&
        receive->packet_snapshot.sideband_kind == packet->sideband_kind &&
        receive->packet_snapshot.sideband_bytes_per_sequence ==
            packet->sideband_bytes_per_sequence;
}

static SparkHiddenGdrPendingReceive *SparkHiddenGdrFindPendingReceive(
    SparkHiddenGdrState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_GDR_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (SparkHiddenGdrPendingReceiveMatchesPacket(
                &state->pending_receives[index], packet) != 0u)
        {
            return &state->pending_receives[index];
        }
    }
    return 0;
}

static SparkHiddenGdrPendingReceive *SparkHiddenGdrReservePendingReceive(
    SparkHiddenGdrState *state)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_GDR_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (state->pending_receives[index].active == 0u)
        {
            memset(&state->pending_receives[index], 0,
                sizeof(state->pending_receives[index]));
            state->pending_receives[index].active = 1u;
            return &state->pending_receives[index];
        }
    }
    return 0;
}

static SparkHiddenGdrRemoteReceive *SparkHiddenGdrFindRemoteReceive(
    SparkHiddenGdrState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_GDR_MAX_REMOTE_RECEIVE_COUNT;
         ++index)
    {
        if (SparkHiddenGdrRemoteReceiveMatchesPacket(
                &state->remote_receives[index], packet) != 0u)
        {
            return &state->remote_receives[index];
        }
    }
    return 0;
}

static SparkStatus SparkHiddenGdrInsertRemoteReceive(
    SparkHiddenGdrState *state,
    const SparkHiddenGdrControlMessage *message)
{
    uint32_t index;
    SparkHiddenGdrRemoteReceive *receive;

    for (index = 0u; index < SPARK_HIDDEN_GDR_MAX_REMOTE_RECEIVE_COUNT;
         ++index)
    {
        if (state->remote_receives[index].active == 0u ||
            state->remote_receives[index].used != 0u)
        {
            receive = &state->remote_receives[index];
            memset(receive, 0, sizeof(*receive));
            receive->active = 1u;
            receive->sequence_id = message->sequence_id;
            receive->token_index = message->token_index;
            receive->active_sequence_count = message->active_sequence_count;
            receive->sideband_kind = message->sideband_kind;
            receive->sideband_bytes_per_sequence =
                message->sideband_bytes_per_sequence;
            receive->hidden_descriptor = message->hidden;
            receive->sideband_descriptor = message->sideband;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_BUSY;
}

static SparkStatus SparkHiddenGdrWriteControlMessage(
    SparkHiddenGdrState *state,
    SparkHiddenGdrControlMessage *message)
{
    if (state == 0 || message == 0 || state->control_fd < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    message->magic = SPARK_HIDDEN_GDR_CONTROL_MAGIC;
    message->version = SPARK_HIDDEN_GDR_CONTROL_VERSION;
    return SparkHiddenGdrWriteFull(state->control_fd, message,
        sizeof(*message));
}

static SparkStatus SparkHiddenGdrReadControlMessageNonblocking(
    SparkHiddenGdrState *state,
    SparkHiddenGdrControlMessage *message)
{
    ssize_t result;
    uint8_t *cursor;
    uint64_t done;

    if (state == 0 || message == 0 || state->control_fd < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    {
        int available_bytes;

        available_bytes = 0;
        if (ioctl(state->control_fd, FIONREAD, &available_bytes) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        if (available_bytes < (int)sizeof(*message))
        {
            return SPARK_STATUS_BUSY;
        }
    }
    cursor = (uint8_t *)message;
    done = 0u;
    while (done < sizeof(*message))
    {
        result = read(state->control_fd, cursor + done,
            (size_t)(sizeof(*message) - done));
        if (result < 0 && errno == EINTR)
        {
            continue;
        }
        if (result <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        done += (uint64_t)result;
    }
    if (message->magic != SPARK_HIDDEN_GDR_CONTROL_MAGIC ||
        message->version != SPARK_HIDDEN_GDR_CONTROL_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenGdrMarkPendingReceiveComplete(
    SparkHiddenGdrState *state,
    const SparkHiddenGdrControlMessage *message)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_GDR_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (state->pending_receives[index].active != 0u &&
            state->pending_receives[index].packet_snapshot.sequence_id ==
                message->sequence_id &&
            state->pending_receives[index].packet_snapshot.token_index ==
                message->token_index)
        {
            state->pending_receives[index].complete = 1u;
            SparkHiddenGdrSignalEvent(state);
            return;
        }
    }
}

static SparkStatus SparkHiddenGdrPumpControl(SparkHiddenGdrState *state)
{
    SparkStatus status;
    SparkHiddenGdrControlMessage message;

    if (state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    while (1)
    {
        memset(&message, 0, sizeof(message));
        status = SparkHiddenGdrReadControlMessageNonblocking(state, &message);
        if (status == SPARK_STATUS_BUSY)
        {
            return SPARK_STATUS_OK;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (message.type == SPARK_HIDDEN_GDR_CONTROL_RECEIVE_READY)
        {
            status = SparkHiddenGdrInsertRemoteReceive(state, &message);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            SparkHiddenGdrSignalEvent(state);
        }
        else if (message.type == SPARK_HIDDEN_GDR_CONTROL_TRANSFER_COMPLETE)
        {
            SparkHiddenGdrMarkPendingReceiveComplete(state, &message);
        }
        else
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
}

static void SparkHiddenGdrBuildCompletion(
    SparkHiddenGdrState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus status)
{
    memset(&state->completion, 0, sizeof(state->completion));
    state->completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    state->completion.descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    state->completion.status = status;
    state->completion.active_sequence_count = packet->active_sequence_count;
    state->completion.sequence_id = packet->sequence_id;
    state->completion.token_index = packet->token_index;
    state->completion.transfer_bytes = SparkHiddenGdrPacketHiddenBytes(packet) +
        SparkHiddenGdrPacketSidebandBytes(packet);
    state->completion.service_time_ns = 0u;
    state->completion_ready = 1u;
    SparkHiddenGdrSignalEvent(state);
}

static SparkStatus SparkHiddenGdrAdvertiseReceive(
    SparkHiddenGdrState *state,
    const SparkHiddenTransportPacket *packet,
    SparkHiddenGdrPendingReceive *receive)
{
    SparkHiddenGdrControlMessage message;
    SparkStatus status;

    if (state == 0 || packet == 0 || receive == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&message, 0, sizeof(message));
    message.type = SPARK_HIDDEN_GDR_CONTROL_RECEIVE_READY;
    message.sequence_id = packet->sequence_id;
    message.token_index = packet->token_index;
    message.active_sequence_count = packet->active_sequence_count;
    message.sideband_kind = packet->sideband_kind;
    message.sideband_bytes_per_sequence = packet->sideband_bytes_per_sequence;
    message.hidden = receive->hidden_descriptor;
    message.sideband = receive->sideband_descriptor;
    status = SparkHiddenGdrWriteControlMessage(state, &message);
    if (status == SPARK_STATUS_OK)
    {
        receive->advertised = 1u;
    }
    return status;
}

static void SparkHiddenGdrReleasePendingReceive(
    SparkHiddenGdrPendingReceive *receive)
{
    if (receive == 0)
    {
        return;
    }
    if (receive->hidden_memory_region != 0)
    {
        ibv_dereg_mr(receive->hidden_memory_region);
    }
    if (receive->sideband_memory_region != 0)
    {
        ibv_dereg_mr(receive->sideband_memory_region);
    }
    memset(receive, 0, sizeof(*receive));
}

static SparkStatus SparkHiddenGdrPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenGdrState *state;
    SparkHiddenGdrPendingReceive *receive;
    SparkStatus status;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;

    state = (SparkHiddenGdrState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&state->endpoint, packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkHiddenGdrDrainEvent(state);
    status = SparkHiddenGdrPumpControl(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    receive = SparkHiddenGdrFindPendingReceive(state, packet);
    if (receive == 0)
    {
        receive = SparkHiddenGdrReservePendingReceive(state);
        if (receive == 0)
        {
            return SPARK_STATUS_BUSY;
        }
        receive->packet_snapshot = *packet;
        hidden_bytes = SparkHiddenGdrPacketHiddenBytes(packet);
        sideband_bytes = SparkHiddenGdrPacketSidebandBytes(packet);
        status = SparkHiddenGdrRegisterReceiveRegion(state,
            packet->hidden_bf16, hidden_bytes,
            &receive->hidden_descriptor, &receive->hidden_memory_region);
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenGdrReleasePendingReceive(receive);
            return status;
        }
        status = SparkHiddenGdrRegisterReceiveRegion(state,
            packet->sideband_payload, sideband_bytes,
            &receive->sideband_descriptor, &receive->sideband_memory_region);
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenGdrReleasePendingReceive(receive);
            return status;
        }
        status = SparkHiddenGdrAdvertiseReceive(state, packet, receive);
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenGdrReleasePendingReceive(receive);
            return status;
        }
    }
    if (receive->complete == 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    SparkHiddenGdrBuildCompletion(state, packet, SPARK_STATUS_OK);
    SparkHiddenGdrReleasePendingReceive(receive);
    return SPARK_STATUS_OK;
}

static void SparkHiddenGdrPartitionRegion(
    uint64_t total_bytes,
    uint32_t lane_count,
    uint32_t lane_index,
    uint64_t *offset_out,
    uint64_t *bytes_out)
{
    uint64_t base;
    uint64_t extra;
    uint64_t offset;
    uint64_t bytes;

    base = total_bytes / (uint64_t)lane_count;
    extra = total_bytes % (uint64_t)lane_count;
    bytes = base + (lane_index < extra ? 1u : 0u);
    offset = base * (uint64_t)lane_index +
        (lane_index < extra ? (uint64_t)lane_index : extra);
    *offset_out = offset;
    *bytes_out = bytes;
}

static SparkStatus SparkHiddenGdrPostWriteForRegion(
    SparkHiddenGdrState *state,
    const void *local_pointer,
    uint64_t local_bytes,
    const SparkHiddenGdrMemoryRegionDescriptor *remote_descriptor,
    struct ibv_mr *local_memory_region,
    uint64_t region_tag,
    uint32_t *posted_write_count_out)
{
    uint32_t lane_index;
    uint32_t posted_write_count;

    if (state == 0 || local_pointer == 0 || local_memory_region == 0 ||
        remote_descriptor == 0 || posted_write_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (local_bytes > remote_descriptor->bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    posted_write_count = 0u;
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        struct ibv_sge scatter_gather;
        struct ibv_send_wr work_request;
        struct ibv_send_wr *bad_work_request;
        uint64_t offset;
        uint64_t bytes;

        SparkHiddenGdrPartitionRegion(local_bytes, state->lane_count,
            lane_index, &offset, &bytes);
        if (bytes == 0u)
        {
            continue;
        }
        memset(&scatter_gather, 0, sizeof(scatter_gather));
        scatter_gather.addr = (uint64_t)(uintptr_t)local_pointer + offset;
        scatter_gather.length = (uint32_t)bytes;
        scatter_gather.lkey = local_memory_region->lkey;
        memset(&work_request, 0, sizeof(work_request));
        work_request.wr_id = ((uint64_t)lane_index << 32) | region_tag;
        work_request.sg_list = &scatter_gather;
        work_request.num_sge = 1;
        work_request.opcode = IBV_WR_RDMA_WRITE;
        work_request.send_flags = IBV_SEND_SIGNALED;
        work_request.wr.rdma.remote_addr = remote_descriptor->address + offset;
        work_request.wr.rdma.rkey = remote_descriptor->rkey;
        bad_work_request = 0;
        if (ibv_post_send(state->lanes[lane_index].queue_pair,
                &work_request, &bad_work_request) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        posted_write_count += 1u;
    }
    *posted_write_count_out += posted_write_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrWaitForPostedWrites(
    SparkHiddenGdrState *state,
    uint32_t expected_write_count)
{
    uint32_t completed_count;
    uint32_t lane_index;
    struct ibv_wc work_completion;
    int result;

    completed_count = 0u;
    while (completed_count < expected_write_count)
    {
        for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
        {
            memset(&work_completion, 0, sizeof(work_completion));
            result = ibv_poll_cq(state->lanes[lane_index].completion_queue,
                1, &work_completion);
            if (result < 0)
            {
                return SPARK_STATUS_IO_ERROR;
            }
            if (result == 0)
            {
                continue;
            }
            if (work_completion.status != IBV_WC_SUCCESS)
            {
                return SPARK_STATUS_IO_ERROR;
            }
            completed_count += 1u;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrSendCompletionMessage(
    SparkHiddenGdrState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus transfer_status)
{
    SparkHiddenGdrControlMessage message;

    memset(&message, 0, sizeof(message));
    message.type = SPARK_HIDDEN_GDR_CONTROL_TRANSFER_COMPLETE;
    message.status = (uint32_t)transfer_status;
    message.sequence_id = packet->sequence_id;
    message.token_index = packet->token_index;
    message.active_sequence_count = packet->active_sequence_count;
    message.sideband_kind = packet->sideband_kind;
    message.sideband_bytes_per_sequence = packet->sideband_bytes_per_sequence;
    return SparkHiddenGdrWriteControlMessage(state, &message);
}

static SparkStatus SparkHiddenGdrSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenGdrState *state;
    SparkHiddenGdrRemoteReceive *remote_receive;
    struct ibv_mr *hidden_memory_region;
    struct ibv_mr *sideband_memory_region;
    SparkStatus status;
    uint32_t posted_write_count;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;

    state = (SparkHiddenGdrState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&state->endpoint, packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkHiddenGdrDrainEvent(state);
    status = SparkHiddenGdrPumpControl(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    remote_receive = SparkHiddenGdrFindRemoteReceive(state, packet);
    if (remote_receive == 0)
    {
        return SPARK_STATUS_BUSY;
    }
    if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    hidden_bytes = SparkHiddenGdrPacketHiddenBytes(packet);
    sideband_bytes = SparkHiddenGdrPacketSidebandBytes(packet);
    status = SparkHiddenGdrGetCachedMemoryRegion(state, packet->hidden_bf16,
        hidden_bytes, &hidden_memory_region);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    sideband_memory_region = 0;
    if (sideband_bytes != 0u)
    {
        status = SparkHiddenGdrGetCachedMemoryRegion(state,
            packet->sideband_payload, sideband_bytes,
            &sideband_memory_region);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    posted_write_count = 0u;
    status = SparkHiddenGdrPostWriteForRegion(state, packet->hidden_bf16,
        hidden_bytes, &remote_receive->hidden_descriptor,
        hidden_memory_region, SPARK_HIDDEN_GDR_WR_ID_REGION_HIDDEN,
        &posted_write_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sideband_bytes != 0u)
    {
        status = SparkHiddenGdrPostWriteForRegion(state,
            packet->sideband_payload, sideband_bytes,
            &remote_receive->sideband_descriptor,
            sideband_memory_region, SPARK_HIDDEN_GDR_WR_ID_REGION_SIDEBAND,
            &posted_write_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    status = SparkHiddenGdrWaitForPostedWrites(state, posted_write_count);
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkHiddenGdrSendCompletionMessage(state, packet, status);
        return status;
    }
    remote_receive->used = 1u;
    status = SparkHiddenGdrSendCompletionMessage(state, packet,
        SPARK_STATUS_OK);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkHiddenGdrBuildCompletion(state, packet, SPARK_STATUS_OK);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    SparkHiddenGdrState *state;
    SparkStatus status;

    state = (SparkHiddenGdrState *)transport_state;
    if (state == 0 || completion == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkHiddenGdrDrainEvent(state);
    status = SparkHiddenGdrPumpControl(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (state->completion_ready == 0u)
    {
        memset(completion, 0, sizeof(*completion));
        completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
        completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
        completion->status = SPARK_STATUS_BUSY;
        return SPARK_STATUS_OK;
    }
    *completion = state->completion;
    state->completion_ready = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrAppendPollDescriptor(
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count,
    int fd,
    uint32_t events)
{
    SparkHiddenTransportPollDescriptor *descriptor;

    if (descriptor_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (fd < 0 || events == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (*descriptor_count >= descriptor_capacity || descriptors == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    descriptor = &descriptors[*descriptor_count];
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    descriptor->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES;
    descriptor->fd = fd;
    descriptor->events = events;
    *descriptor_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrGetPollDescriptors(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out)
{
    SparkHiddenGdrState *state;
    uint32_t descriptor_count;
    SparkStatus status;

    if (transport_state == 0 || descriptor_count_out == 0 ||
        (descriptors == 0 && descriptor_capacity != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenGdrState *)transport_state;
    descriptor_count = 0u;
    status = SparkHiddenGdrAppendPollDescriptor(descriptors,
        descriptor_capacity, &descriptor_count, state->event_fd,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkHiddenGdrAppendPollDescriptor(descriptors,
        descriptor_capacity, &descriptor_count, state->control_fd,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *descriptor_count_out = descriptor_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    uint32_t packet_index;
    SparkStatus status;

    if (packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenGdrSend(transport_state, &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenGdrPostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    uint32_t packet_index;
    SparkStatus status;

    if (packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenGdrPostReceive(transport_state,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        {
            return status;
        }
        if (status == SPARK_STATUS_BUSY)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenGdrDestroyState(SparkHiddenGdrState *state)
{
    uint32_t lane_index;
    uint32_t receive_index;

    if (state == 0)
    {
        return;
    }
    SparkHiddenGdrDeregisterCachedMemoryRegions(state);
    for (receive_index = 0u;
         receive_index < SPARK_HIDDEN_GDR_MAX_PENDING_RECEIVE_COUNT;
         ++receive_index)
    {
        SparkHiddenGdrReleasePendingReceive(&state->pending_receives[receive_index]);
    }
    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        if (state->lanes[lane_index].queue_pair != 0)
        {
            ibv_destroy_qp(state->lanes[lane_index].queue_pair);
        }
        if (state->lanes[lane_index].completion_queue != 0)
        {
            ibv_destroy_cq(state->lanes[lane_index].completion_queue);
        }
    }
    if (state->protection_domain != 0)
    {
        ibv_dealloc_pd(state->protection_domain);
    }
    if (state->verbs_context != 0)
    {
        ibv_close_device(state->verbs_context);
    }
    state->control_fd = SparkHiddenGdrCloseFd(state->control_fd);
    state->listen_fd = SparkHiddenGdrCloseFd(state->listen_fd);
    state->event_fd = SparkHiddenGdrCloseFd(state->event_fd);
    free(state);
}

static SparkStatus SparkHiddenGdrInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenGdrState *state;
    SparkStatus status;
    uint32_t lane_count;
    const char *rank_text;

    if (endpoint == 0 || transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidateZeroCopyGpudirectEndpoint(endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    state = (SparkHiddenGdrState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->endpoint = *endpoint;
    state->listen_fd = -1;
    state->control_fd = -1;
    state->event_fd = -1;
    state->debug_enabled = getenv("SPARKPIPE_HIDDEN_GDR_DEBUG") != 0 ?
        1u : 0u;
    lane_count = SparkHiddenGdrParseUintEnv("SPARKPIPE_HIDDEN_GDR_LANES",
        SPARK_HIDDEN_GDR_DEFAULT_LANE_COUNT);
    if (lane_count == 0u)
    {
        lane_count = 1u;
    }
    if (lane_count > SPARK_HIDDEN_GDR_MAX_LANE_COUNT)
    {
        lane_count = SPARK_HIDDEN_GDR_MAX_LANE_COUNT;
    }
    state->lane_count = lane_count;
    state->control_port_base = SparkHiddenGdrParseUintEnv(
        "SPARKPIPE_HIDDEN_GDR_CONTROL_PORT_BASE",
        SPARK_HIDDEN_GDR_DEFAULT_CONTROL_PORT_BASE);
    state->verbs_port = (uint8_t)SparkHiddenGdrParseUintEnv(
        "SPARKPIPE_HIDDEN_GDR_IB_PORT", 1u);
    state->gid_index = SparkHiddenGdrParseIntEnv(
        "SPARKPIPE_HIDDEN_GDR_GID_INDEX", 0);
    rank_text = getenv("SPARKPIPE_PP13_TRANSPORT_RANK");
    if (rank_text == 0 || rank_text[0] == '\0')
    {
        SparkHiddenGdrDestroyState(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->local_rank = (int32_t)SparkHiddenGdrParseUintEnv(
        "SPARKPIPE_PP13_TRANSPORT_RANK", 1000u);
    status = SparkHiddenGdrParseRoute(endpoint->route_name,
        state->source_host, state->sink_host);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenGdrDestroyState(state);
        return status;
    }
    state->source_rank = SparkHiddenGdrRankFromHost(state->source_host);
    state->sink_rank = SparkHiddenGdrRankFromHost(state->sink_host);
    if (state->source_rank < 0 || state->sink_rank < 0)
    {
        SparkHiddenGdrDestroyState(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->is_sender = state->local_rank == state->source_rank ? 1u : 0u;
    if (state->is_sender == 0u && state->local_rank != state->sink_rank)
    {
        SparkHiddenGdrDestroyState(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->event_fd = eventfd(0u, EFD_NONBLOCK | EFD_CLOEXEC);
    if (state->event_fd < 0)
    {
        SparkHiddenGdrDestroyState(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkHiddenGdrOpenVerbsDevice(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenGdrDestroyState(state);
        return status;
    }
    status = SparkHiddenGdrCreateQueuePairs(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenGdrDestroyState(state);
        return status;
    }
    status = SparkHiddenGdrConnectControl(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenGdrDestroyState(state);
        return status;
    }
    status = SparkHiddenGdrExchangeQueuePairInfo(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenGdrDestroyState(state);
        return status;
    }
    status = SparkHiddenGdrReadyQueuePairs(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenGdrDestroyState(state);
        return status;
    }
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_gdr_ready route=%s rank=%d sender=%u lanes=%u\n",
            endpoint->route_name,
            state->local_rank,
            state->is_sender,
            state->lane_count);
    }
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenGdrDestroy(void *transport_state)
{
    SparkHiddenGdrDestroyState((SparkHiddenGdrState *)transport_state);
}

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface transport_interface;

    memset(&transport_interface, 0, sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_ZERO_COPY_GPUDIRECT_CAPS;
    transport_interface.initialize = SparkHiddenGdrInitialize;
    transport_interface.destroy = SparkHiddenGdrDestroy;
    transport_interface.post_receive = SparkHiddenGdrPostReceive;
    transport_interface.send = SparkHiddenGdrSend;
    transport_interface.poll = SparkHiddenGdrPoll;
    transport_interface.post_receive_batch = SparkHiddenGdrPostReceiveBatch;
    transport_interface.send_batch = SparkHiddenGdrSendBatch;
    transport_interface.get_poll_descriptors = SparkHiddenGdrGetPollDescriptors;
    return &transport_interface;
}

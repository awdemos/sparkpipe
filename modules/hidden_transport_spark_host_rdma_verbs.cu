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

#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_MAGIC 0x53475055u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION 1u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_CONTROL_PORT_BASE 55700u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_LANE_COUNT 8u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT 32u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS 50u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES 64u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_POLL_TIMEOUT_MS 1000u

#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HELLO 1u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY 2u
#define SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_TRANSFER_COMPLETE 3u

#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_REGION_HIDDEN 1ull
#define SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_REGION_SIDEBAND 2ull

#define SPARK_HIDDEN_SPARK_HOST_RDMA_NO_INDEX 0xffffffffu

typedef struct SparkHiddenSparkHostRdmaMemoryRegionDescriptor
{
    uint64_t address;
    uint64_t bytes;
    uint32_t rkey;
    uint32_t reserved;
} SparkHiddenSparkHostRdmaMemoryRegionDescriptor;

typedef struct SparkHiddenSparkHostRdmaControlMessage
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
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor hidden;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor sideband;
} SparkHiddenSparkHostRdmaControlMessage;

typedef struct SparkHiddenSparkHostRdmaQueuePairWireInfo
{
    uint32_t qp_number;
    uint32_t packet_sequence_number;
    uint16_t lid;
    uint16_t reserved;
    uint8_t gid[16u];
} SparkHiddenSparkHostRdmaQueuePairWireInfo;

typedef struct SparkHiddenSparkHostRdmaLane
{
    struct ibv_cq *completion_queue;
    struct ibv_qp *queue_pair;
    SparkHiddenSparkHostRdmaQueuePairWireInfo local_info;
    SparkHiddenSparkHostRdmaQueuePairWireInfo remote_info;
} SparkHiddenSparkHostRdmaLane;

typedef struct SparkHiddenSparkHostRdmaCachedMemoryRegion
{
    const void *pointer;
    uint64_t bytes;
    uint64_t last_use_epoch;
    struct ibv_mr *memory_region;
} SparkHiddenSparkHostRdmaCachedMemoryRegion;

typedef struct SparkHiddenSparkHostRdmaPendingReceive
{
    uint32_t active;
    uint32_t complete;
    uint32_t advertised;
    SparkHiddenTransportPacket packet_snapshot;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor hidden_descriptor;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor sideband_descriptor;
    struct ibv_mr *hidden_memory_region;
    struct ibv_mr *sideband_memory_region;
} SparkHiddenSparkHostRdmaPendingReceive;

typedef struct SparkHiddenSparkHostRdmaRemoteReceive
{
    uint32_t active;
    uint32_t used;
    uint64_t sequence_id;
    uint64_t token_index;
    uint32_t active_sequence_count;
    uint32_t sideband_kind;
    uint32_t sideband_bytes_per_sequence;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor hidden_descriptor;
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor sideband_descriptor;
} SparkHiddenSparkHostRdmaRemoteReceive;

typedef struct SparkHiddenSparkHostRdmaState
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
    char source_host[SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES];
    char sink_host[SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES];
    struct ibv_context *verbs_context;
    struct ibv_pd *protection_domain;
    union ibv_gid local_gid;
    uint16_t local_lid;
    SparkHiddenSparkHostRdmaLane lanes[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    SparkHiddenSparkHostRdmaCachedMemoryRegion cached_regions[SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT];
    uint64_t memory_region_epoch;
    SparkHiddenSparkHostRdmaPendingReceive pending_receives[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT];
    SparkHiddenSparkHostRdmaRemoteReceive remote_receives[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT];
    SparkHiddenTransportCompletion completion;
    uint32_t completion_ready;
} SparkHiddenSparkHostRdmaState;

static uint32_t SparkHiddenSparkHostRdmaParseUintEnv(
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

static int32_t SparkHiddenSparkHostRdmaParseIntEnv(
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

static int SparkHiddenSparkHostRdmaCloseFd(int fd)
{
    if (fd >= 0)
    {
        (void)close(fd);
    }
    return -1;
}

static void SparkHiddenSparkHostRdmaSignalEvent(SparkHiddenSparkHostRdmaState *state)
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

static void SparkHiddenSparkHostRdmaDrainEvent(SparkHiddenSparkHostRdmaState *state)
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

static SparkStatus SparkHiddenSparkHostRdmaSetNonblocking(int fd)
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

static void SparkHiddenSparkHostRdmaConfigureSocket(int fd)
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

static int SparkHiddenSparkHostRdmaListen(uint32_t port)
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

static SparkStatus SparkHiddenSparkHostRdmaReadFull(
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

static SparkStatus SparkHiddenSparkHostRdmaWriteFull(
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

static int32_t SparkHiddenSparkHostRdmaRankFromHost(const char *host)
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

static SparkStatus SparkHiddenSparkHostRdmaParseRoute(
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
        source_bytes >= SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES ||
        sink_bytes >= SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_HOST_BYTES)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(source_host, route_name, (size_t)source_bytes);
    source_host[source_bytes] = '\0';
    memcpy(sink_host, middle + 4, (size_t)sink_bytes);
    sink_host[sink_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaConnectControl(
    SparkHiddenSparkHostRdmaState *state)
{
    struct addrinfo hints;
    struct addrinfo *result;
    struct addrinfo *entry;
    char port_text[16u];
    int fd;
    const char *host;

    if (state->is_sender == 0u)
    {
        state->listen_fd = SparkHiddenSparkHostRdmaListen(
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
        SparkHiddenSparkHostRdmaConfigureSocket(state->control_fd);
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
            (void)poll(0, 0, SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS);
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
                SparkHiddenSparkHostRdmaConfigureSocket(state->control_fd);
            }
            else
            {
                (void)close(fd);
            }
        }
        freeaddrinfo(result);
        if (state->control_fd < 0)
        {
            (void)poll(0, 0, SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS);
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaOpenVerbsDevice(SparkHiddenSparkHostRdmaState *state)
{
    struct ibv_device **devices;
    struct ibv_device *selected_device;
    const char *requested_name;
    int count;
    int index;
    struct ibv_port_attr port_attributes;

    requested_name = getenv("SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_DEVICE");
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

static SparkStatus SparkHiddenSparkHostRdmaModifyQueuePairToInit(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaLane *lane)
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

static SparkStatus SparkHiddenSparkHostRdmaCreateQueuePairs(SparkHiddenSparkHostRdmaState *state)
{
    struct ibv_qp_init_attr init_attributes;
    uint32_t lane_index;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        SparkHiddenSparkHostRdmaLane *lane;

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
        if (SparkHiddenSparkHostRdmaModifyQueuePairToInit(state, lane) !=
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

static SparkStatus SparkHiddenSparkHostRdmaExchangeQueuePairInfo(
    SparkHiddenSparkHostRdmaState *state)
{
    SparkHiddenSparkHostRdmaQueuePairWireInfo local_infos[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
    SparkHiddenSparkHostRdmaQueuePairWireInfo remote_infos[SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT];
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
        (uint32_t)sizeof(SparkHiddenSparkHostRdmaQueuePairWireInfo);
    if (state->is_sender != 0u)
    {
        status = SparkHiddenSparkHostRdmaWriteFull(state->control_fd,
            &state->lane_count, sizeof(state->lane_count));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenSparkHostRdmaWriteFull(state->control_fd,
            local_infos, bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenSparkHostRdmaReadFull(state->control_fd,
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
        status = SparkHiddenSparkHostRdmaReadFull(state->control_fd,
            &remote_lane_count, sizeof(remote_lane_count));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (remote_lane_count != state->lane_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkHiddenSparkHostRdmaReadFull(state->control_fd,
            remote_infos, bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkHiddenSparkHostRdmaWriteFull(state->control_fd,
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
    return SparkHiddenSparkHostRdmaSetNonblocking(state->control_fd);
}

static SparkStatus SparkHiddenSparkHostRdmaModifyQueuePairToReady(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaLane *lane)
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

static SparkStatus SparkHiddenSparkHostRdmaReadyQueuePairs(SparkHiddenSparkHostRdmaState *state)
{
    uint32_t lane_index;
    SparkStatus status;

    for (lane_index = 0u; lane_index < state->lane_count; ++lane_index)
    {
        status = SparkHiddenSparkHostRdmaModifyQueuePairToReady(state,
            &state->lanes[lane_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDeregisterCachedMemoryRegions(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region != 0)
        {
            ibv_dereg_mr(state->cached_regions[index].memory_region);
        }
        memset(&state->cached_regions[index], 0,
            sizeof(state->cached_regions[index]));
    }
}


static SparkStatus SparkHiddenSparkHostRdmaResolveMappedHostPointer(
    const void *cuda_visible_pointer,
    uint64_t bytes,
    void **host_pointer_out)
{
    cudaPointerAttributes attributes;
    cudaError_t cuda_status;

    if (cuda_visible_pointer == 0 || bytes == 0u || host_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *host_pointer_out = 0;
    memset(&attributes, 0, sizeof(attributes));
    cuda_status = cudaPointerGetAttributes(&attributes, cuda_visible_pointer);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
    if (attributes.hostPointer == 0 || attributes.devicePointer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *host_pointer_out = attributes.hostPointer;
#else
    if (attributes.memoryType != cudaMemoryTypeHost)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *host_pointer_out = (void *)cuda_visible_pointer;
#endif
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaGetCachedMemoryRegion(
    SparkHiddenSparkHostRdmaState *state,
    const void *pointer,
    uint64_t bytes,
    struct ibv_mr **memory_region_out,
    void **registered_host_pointer_out)
{
    uint32_t index;
    uint32_t best_index;
    uint64_t best_epoch;
    int access_flags;
    struct ibv_mr *memory_region;
    void *host_pointer;
    SparkStatus status;

    if (state == 0 || pointer == 0 || bytes == 0u ||
        memory_region_out == 0 || registered_host_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenSparkHostRdmaResolveMappedHostPointer(
        pointer, bytes, &host_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *registered_host_pointer_out = host_pointer;
    state->memory_region_epoch += 1u;
    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT; ++index)
    {
        if (state->cached_regions[index].memory_region != 0 &&
            state->cached_regions[index].pointer == host_pointer &&
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
    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MR_CACHE_COUNT; ++index)
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
    memory_region = ibv_reg_mr(state->protection_domain, host_pointer,
        (size_t)bytes, access_flags);
    if (memory_region == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    state->cached_regions[best_index].pointer = host_pointer;
    state->cached_regions[best_index].bytes = bytes;
    state->cached_regions[best_index].last_use_epoch =
        state->memory_region_epoch;
    state->cached_regions[best_index].memory_region = memory_region;
    *memory_region_out = memory_region;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaRegisterReceiveRegion(
    SparkHiddenSparkHostRdmaState *state,
    const void *pointer,
    uint64_t bytes,
    SparkHiddenSparkHostRdmaMemoryRegionDescriptor *descriptor,
    struct ibv_mr **memory_region_out)
{
    int access_flags;
    struct ibv_mr *memory_region;
    void *host_pointer;
    SparkStatus status;

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
    status = SparkHiddenSparkHostRdmaResolveMappedHostPointer(
        pointer, bytes, &host_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    memory_region = ibv_reg_mr(state->protection_domain, host_pointer,
        (size_t)bytes, access_flags);
    if (memory_region == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    descriptor->address = (uint64_t)(uintptr_t)host_pointer;
    descriptor->bytes = bytes;
    descriptor->rkey = memory_region->rkey;
    *memory_region_out = memory_region;
    return SPARK_STATUS_OK;
}

static uint64_t SparkHiddenSparkHostRdmaPacketHiddenBytes(
    const SparkHiddenTransportPacket *packet)
{
    return (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
}

static uint64_t SparkHiddenSparkHostRdmaPacketSidebandBytes(
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

static uint32_t SparkHiddenSparkHostRdmaRemoteReceiveMatchesPacket(
    const SparkHiddenSparkHostRdmaRemoteReceive *receive,
    const SparkHiddenTransportPacket *packet)
{
    return receive != 0 && packet != 0 && receive->active != 0u &&
        receive->used == 0u && receive->sequence_id == packet->sequence_id &&
        receive->token_index == packet->token_index &&
        receive->active_sequence_count == packet->active_sequence_count &&
        receive->sideband_kind == packet->sideband_kind &&
        receive->sideband_bytes_per_sequence == packet->sideband_bytes_per_sequence;
}

static uint32_t SparkHiddenSparkHostRdmaPendingReceiveMatchesPacket(
    const SparkHiddenSparkHostRdmaPendingReceive *receive,
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

static SparkHiddenSparkHostRdmaPendingReceive *SparkHiddenSparkHostRdmaFindPendingReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (SparkHiddenSparkHostRdmaPendingReceiveMatchesPacket(
                &state->pending_receives[index], packet) != 0u)
        {
            return &state->pending_receives[index];
        }
    }
    return 0;
}

static SparkHiddenSparkHostRdmaPendingReceive *SparkHiddenSparkHostRdmaReservePendingReceive(
    SparkHiddenSparkHostRdmaState *state)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
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

static SparkHiddenSparkHostRdmaRemoteReceive *SparkHiddenSparkHostRdmaFindRemoteReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT;
         ++index)
    {
        if (SparkHiddenSparkHostRdmaRemoteReceiveMatchesPacket(
                &state->remote_receives[index], packet) != 0u)
        {
            return &state->remote_receives[index];
        }
    }
    return 0;
}

static SparkStatus SparkHiddenSparkHostRdmaInsertRemoteReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    uint32_t index;
    SparkHiddenSparkHostRdmaRemoteReceive *receive;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_REMOTE_RECEIVE_COUNT;
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

static SparkStatus SparkHiddenSparkHostRdmaWriteControlMessage(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaControlMessage *message)
{
    if (state == 0 || message == 0 || state->control_fd < 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    message->magic = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_MAGIC;
    message->version = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION;
    return SparkHiddenSparkHostRdmaWriteFull(state->control_fd, message,
        sizeof(*message));
}

static SparkStatus SparkHiddenSparkHostRdmaReadControlMessageNonblocking(
    SparkHiddenSparkHostRdmaState *state,
    SparkHiddenSparkHostRdmaControlMessage *message)
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
    if (message->magic != SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_MAGIC ||
        message->version != SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_VERSION)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaMarkPendingReceiveComplete(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenSparkHostRdmaControlMessage *message)
{
    uint32_t index;

    for (index = 0u; index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++index)
    {
        if (state->pending_receives[index].active != 0u &&
            state->pending_receives[index].packet_snapshot.sequence_id ==
                message->sequence_id &&
            state->pending_receives[index].packet_snapshot.token_index ==
                message->token_index)
        {
            state->pending_receives[index].complete = 1u;
            SparkHiddenSparkHostRdmaSignalEvent(state);
            return;
        }
    }
}

static SparkStatus SparkHiddenSparkHostRdmaPumpControl(SparkHiddenSparkHostRdmaState *state)
{
    SparkStatus status;
    SparkHiddenSparkHostRdmaControlMessage message;

    if (state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    while (1)
    {
        memset(&message, 0, sizeof(message));
        status = SparkHiddenSparkHostRdmaReadControlMessageNonblocking(state, &message);
        if (status == SPARK_STATUS_BUSY)
        {
            return SPARK_STATUS_OK;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (message.type == SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY)
        {
            status = SparkHiddenSparkHostRdmaInsertRemoteReceive(state, &message);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            SparkHiddenSparkHostRdmaSignalEvent(state);
        }
        else if (message.type == SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_TRANSFER_COMPLETE)
        {
            SparkHiddenSparkHostRdmaMarkPendingReceiveComplete(state, &message);
        }
        else
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
}

static void SparkHiddenSparkHostRdmaBuildCompletion(
    SparkHiddenSparkHostRdmaState *state,
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
    state->completion.transfer_bytes = SparkHiddenSparkHostRdmaPacketHiddenBytes(packet) +
        SparkHiddenSparkHostRdmaPacketSidebandBytes(packet);
    state->completion.service_time_ns = 0u;
    state->completion_ready = 1u;
    SparkHiddenSparkHostRdmaSignalEvent(state);
}

static SparkStatus SparkHiddenSparkHostRdmaAdvertiseReceive(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    SparkHiddenSparkHostRdmaPendingReceive *receive)
{
    SparkHiddenSparkHostRdmaControlMessage message;
    SparkStatus status;

    if (state == 0 || packet == 0 || receive == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&message, 0, sizeof(message));
    message.type = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY;
    message.sequence_id = packet->sequence_id;
    message.token_index = packet->token_index;
    message.active_sequence_count = packet->active_sequence_count;
    message.sideband_kind = packet->sideband_kind;
    message.sideband_bytes_per_sequence = packet->sideband_bytes_per_sequence;
    message.hidden = receive->hidden_descriptor;
    message.sideband = receive->sideband_descriptor;
    status = SparkHiddenSparkHostRdmaWriteControlMessage(state, &message);
    if (status == SPARK_STATUS_OK)
    {
        receive->advertised = 1u;
    }
    return status;
}

static void SparkHiddenSparkHostRdmaReleasePendingReceive(
    SparkHiddenSparkHostRdmaPendingReceive *receive)
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

static SparkStatus SparkHiddenSparkHostRdmaPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaPendingReceive *receive;
    SparkStatus status;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&state->endpoint, packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkHiddenSparkHostRdmaDrainEvent(state);
    status = SparkHiddenSparkHostRdmaPumpControl(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    receive = SparkHiddenSparkHostRdmaFindPendingReceive(state, packet);
    if (receive == 0)
    {
        receive = SparkHiddenSparkHostRdmaReservePendingReceive(state);
        if (receive == 0)
        {
            return SPARK_STATUS_BUSY;
        }
        receive->packet_snapshot = *packet;
        hidden_bytes = SparkHiddenSparkHostRdmaPacketHiddenBytes(packet);
        sideband_bytes = SparkHiddenSparkHostRdmaPacketSidebandBytes(packet);
        status = SparkHiddenSparkHostRdmaRegisterReceiveRegion(state,
            packet->hidden_bf16, hidden_bytes,
            &receive->hidden_descriptor, &receive->hidden_memory_region);
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenSparkHostRdmaReleasePendingReceive(receive);
            return status;
        }
        status = SparkHiddenSparkHostRdmaRegisterReceiveRegion(state,
            packet->sideband_payload, sideband_bytes,
            &receive->sideband_descriptor, &receive->sideband_memory_region);
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenSparkHostRdmaReleasePendingReceive(receive);
            return status;
        }
        status = SparkHiddenSparkHostRdmaAdvertiseReceive(state, packet, receive);
        if (status != SPARK_STATUS_OK)
        {
            SparkHiddenSparkHostRdmaReleasePendingReceive(receive);
            return status;
        }
    }
    if (receive->complete == 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    SparkHiddenSparkHostRdmaBuildCompletion(state, packet, SPARK_STATUS_OK);
    SparkHiddenSparkHostRdmaReleasePendingReceive(receive);
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaPartitionRegion(
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

static SparkStatus SparkHiddenSparkHostRdmaPostWriteForRegion(
    SparkHiddenSparkHostRdmaState *state,
    const void *local_pointer,
    uint64_t local_bytes,
    const SparkHiddenSparkHostRdmaMemoryRegionDescriptor *remote_descriptor,
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

        SparkHiddenSparkHostRdmaPartitionRegion(local_bytes, state->lane_count,
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

static SparkStatus SparkHiddenSparkHostRdmaWaitForPostedWrites(
    SparkHiddenSparkHostRdmaState *state,
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

static SparkStatus SparkHiddenSparkHostRdmaSendCompletionMessage(
    SparkHiddenSparkHostRdmaState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus transfer_status)
{
    SparkHiddenSparkHostRdmaControlMessage message;

    memset(&message, 0, sizeof(message));
    message.type = SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_TRANSFER_COMPLETE;
    message.status = (uint32_t)transfer_status;
    message.sequence_id = packet->sequence_id;
    message.token_index = packet->token_index;
    message.active_sequence_count = packet->active_sequence_count;
    message.sideband_kind = packet->sideband_kind;
    message.sideband_bytes_per_sequence = packet->sideband_bytes_per_sequence;
    return SparkHiddenSparkHostRdmaWriteControlMessage(state, &message);
}

static SparkStatus SparkHiddenSparkHostRdmaSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkHiddenSparkHostRdmaRemoteReceive *remote_receive;
    struct ibv_mr *hidden_memory_region;
    struct ibv_mr *sideband_memory_region;
    void *hidden_host_pointer;
    void *sideband_host_pointer;
    SparkStatus status;
    uint32_t posted_write_count;
    uint64_t hidden_bytes;
    uint64_t sideband_bytes;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || packet == 0 || state->is_sender == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&state->endpoint, packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkHiddenSparkHostRdmaDrainEvent(state);
    status = SparkHiddenSparkHostRdmaPumpControl(state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    remote_receive = SparkHiddenSparkHostRdmaFindRemoteReceive(state, packet);
    if (remote_receive == 0)
    {
        return SPARK_STATUS_BUSY;
    }
    if (cudaStreamSynchronize((cudaStream_t)packet->cuda_stream) != cudaSuccess)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    hidden_bytes = SparkHiddenSparkHostRdmaPacketHiddenBytes(packet);
    sideband_bytes = SparkHiddenSparkHostRdmaPacketSidebandBytes(packet);
    status = SparkHiddenSparkHostRdmaGetCachedMemoryRegion(state, packet->hidden_bf16,
        hidden_bytes, &hidden_memory_region, &hidden_host_pointer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    sideband_memory_region = 0;
    if (sideband_bytes != 0u)
    {
        status = SparkHiddenSparkHostRdmaGetCachedMemoryRegion(state,
            packet->sideband_payload, sideband_bytes,
            &sideband_memory_region, &sideband_host_pointer);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    posted_write_count = 0u;
    status = SparkHiddenSparkHostRdmaPostWriteForRegion(state, hidden_host_pointer,
        hidden_bytes, &remote_receive->hidden_descriptor,
        hidden_memory_region, SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_REGION_HIDDEN,
        &posted_write_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sideband_bytes != 0u)
    {
        status = SparkHiddenSparkHostRdmaPostWriteForRegion(state,
            sideband_host_pointer, sideband_bytes,
            &remote_receive->sideband_descriptor,
            sideband_memory_region, SPARK_HIDDEN_SPARK_HOST_RDMA_WR_ID_REGION_SIDEBAND,
            &posted_write_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    status = SparkHiddenSparkHostRdmaWaitForPostedWrites(state, posted_write_count);
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkHiddenSparkHostRdmaSendCompletionMessage(state, packet, status);
        return status;
    }
    remote_receive->used = 1u;
    status = SparkHiddenSparkHostRdmaSendCompletionMessage(state, packet,
        SPARK_STATUS_OK);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkHiddenSparkHostRdmaBuildCompletion(state, packet, SPARK_STATUS_OK);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;

    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    if (state == 0 || completion == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkHiddenSparkHostRdmaDrainEvent(state);
    status = SparkHiddenSparkHostRdmaPumpControl(state);
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

static SparkStatus SparkHiddenSparkHostRdmaAppendPollDescriptor(
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

static SparkStatus SparkHiddenSparkHostRdmaGetPollDescriptors(
    void *transport_state,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out)
{
    SparkHiddenSparkHostRdmaState *state;
    uint32_t descriptor_count;
    SparkStatus status;

    if (transport_state == 0 || descriptor_count_out == 0 ||
        (descriptors == 0 && descriptor_capacity != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenSparkHostRdmaState *)transport_state;
    descriptor_count = 0u;
    status = SparkHiddenSparkHostRdmaAppendPollDescriptor(descriptors,
        descriptor_capacity, &descriptor_count, state->event_fd,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkHiddenSparkHostRdmaAppendPollDescriptor(descriptors,
        descriptor_capacity, &descriptor_count, state->control_fd,
        SPARK_HIDDEN_TRANSPORT_POLL_READ);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *descriptor_count_out = descriptor_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaSendBatch(
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
        status = SparkHiddenSparkHostRdmaSend(transport_state, &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenSparkHostRdmaPostReceiveBatch(
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
        status = SparkHiddenSparkHostRdmaPostReceive(transport_state,
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

static void SparkHiddenSparkHostRdmaDestroyState(SparkHiddenSparkHostRdmaState *state)
{
    uint32_t lane_index;
    uint32_t receive_index;

    if (state == 0)
    {
        return;
    }
    SparkHiddenSparkHostRdmaDeregisterCachedMemoryRegions(state);
    for (receive_index = 0u;
         receive_index < SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_PENDING_RECEIVE_COUNT;
         ++receive_index)
    {
        SparkHiddenSparkHostRdmaReleasePendingReceive(&state->pending_receives[receive_index]);
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
    state->control_fd = SparkHiddenSparkHostRdmaCloseFd(state->control_fd);
    state->listen_fd = SparkHiddenSparkHostRdmaCloseFd(state->listen_fd);
    state->event_fd = SparkHiddenSparkHostRdmaCloseFd(state->event_fd);
    free(state);
}

static SparkStatus SparkHiddenSparkHostRdmaInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenSparkHostRdmaState *state;
    SparkStatus status;
    uint32_t lane_count;
    const char *rank_text;

    if (endpoint == 0 || transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidateSparkHostRdmaEndpoint(endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    state = (SparkHiddenSparkHostRdmaState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->endpoint = *endpoint;
    state->listen_fd = -1;
    state->control_fd = -1;
    state->event_fd = -1;
    state->debug_enabled = getenv("SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DEBUG") != 0 ?
        1u : 0u;
    lane_count = SparkHiddenSparkHostRdmaParseUintEnv("SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_LANES",
        SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_LANE_COUNT);
    if (lane_count == 0u)
    {
        lane_count = 1u;
    }
    if (lane_count > SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT)
    {
        lane_count = SPARK_HIDDEN_SPARK_HOST_RDMA_MAX_LANE_COUNT;
    }
    state->lane_count = lane_count;
    state->control_port_base = SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_CONTROL_PORT_BASE",
        SPARK_HIDDEN_SPARK_HOST_RDMA_DEFAULT_CONTROL_PORT_BASE);
    state->verbs_port = (uint8_t)SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_PORT", 1u);
    state->gid_index = SparkHiddenSparkHostRdmaParseIntEnv(
        "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_GID_INDEX", 0);
    rank_text = getenv("SPARKPIPE_PP13_TRANSPORT_RANK");
    if (rank_text == 0 || rank_text[0] == '\0')
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state->local_rank = (int32_t)SparkHiddenSparkHostRdmaParseUintEnv(
        "SPARKPIPE_PP13_TRANSPORT_RANK", 1000u);
    status = SparkHiddenSparkHostRdmaParseRoute(endpoint->route_name,
        state->source_host, state->sink_host);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    state->source_rank = SparkHiddenSparkHostRdmaRankFromHost(state->source_host);
    state->sink_rank = SparkHiddenSparkHostRdmaRankFromHost(state->sink_host);
    if (state->source_rank < 0 || state->sink_rank < 0)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->is_sender = state->local_rank == state->source_rank ? 1u : 0u;
    if (state->is_sender == 0u && state->local_rank != state->sink_rank)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    state->event_fd = eventfd(0u, EFD_NONBLOCK | EFD_CLOEXEC);
    if (state->event_fd < 0)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkHiddenSparkHostRdmaOpenVerbsDevice(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaCreateQueuePairs(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaConnectControl(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaExchangeQueuePairInfo(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    status = SparkHiddenSparkHostRdmaReadyQueuePairs(state);
    if (status != SPARK_STATUS_OK)
    {
        SparkHiddenSparkHostRdmaDestroyState(state);
        return status;
    }
    if (state->debug_enabled != 0u)
    {
        fprintf(stderr,
            "hidden_spark_host_rdma_ready route=%s rank=%d sender=%u lanes=%u\n",
            endpoint->route_name,
            state->local_rank,
            state->is_sender,
            state->lane_count);
    }
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenSparkHostRdmaDestroy(void *transport_state)
{
    SparkHiddenSparkHostRdmaDestroyState((SparkHiddenSparkHostRdmaState *)transport_state);
}

extern "C" const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface transport_interface;

    memset(&transport_interface, 0, sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS;
    transport_interface.initialize = SparkHiddenSparkHostRdmaInitialize;
    transport_interface.destroy = SparkHiddenSparkHostRdmaDestroy;
    transport_interface.post_receive = SparkHiddenSparkHostRdmaPostReceive;
    transport_interface.send = SparkHiddenSparkHostRdmaSend;
    transport_interface.poll = SparkHiddenSparkHostRdmaPoll;
    transport_interface.post_receive_batch = SparkHiddenSparkHostRdmaPostReceiveBatch;
    transport_interface.send_batch = SparkHiddenSparkHostRdmaSendBatch;
    transport_interface.get_poll_descriptors = SparkHiddenSparkHostRdmaGetPollDescriptors;
    return &transport_interface;
}

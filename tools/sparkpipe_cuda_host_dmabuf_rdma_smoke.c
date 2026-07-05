#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>
#include <infiniband/verbs.h>

typedef struct SparkDmabufRdmaPeerInfo
{
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    uint64_t addr;
    union ibv_gid gid;
} SparkDmabufRdmaPeerInfo;

typedef struct SparkDmabufRdmaResources
{
    struct ibv_context *verbs_context;
    struct ibv_pd *protection_domain;
    struct ibv_cq *completion_queue;
    struct ibv_qp *queue_pair;
    struct ibv_mr *memory_region;
    CUcontext cuda_context;
    CUdeviceptr device_buffer;
    void *host_buffer;
    int32_t dmabuf_fd;
    uint32_t psn;
    union ibv_gid gid;
    size_t bytes;
} SparkDmabufRdmaResources;

static int32_t SparkDmabufRdmaTcpListen(int32_t port)
{
    struct sockaddr_in address;
    int32_t fd;
    int32_t opt;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return(-1);
    opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
        return(-2);
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        return(-3);
    if (listen(fd, 1) < 0)
        return(-4);
    return(fd);
}

static int32_t SparkDmabufRdmaTcpConnect(const char *host,int32_t port)
{
    struct sockaddr_in address;
    int32_t fd;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return(-1);
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1)
        return(-2);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) < 0)
        return(-3);
    return(fd);
}

static int32_t SparkDmabufRdmaReadFull(int32_t fd,void *buffer,size_t bytes)
{
    uint8_t *p;
    ssize_t n;
    size_t done;

    p = (uint8_t *)buffer;
    done = 0u;
    while (done < bytes)
    {
        n = read(fd, &p[done], (bytes - done));
        if (n <= 0)
            return(-1);
        done += (size_t)n;
    }
    return(0);
}

static int32_t SparkDmabufRdmaWriteFull(int32_t fd,const void *buffer,size_t bytes)
{
    const uint8_t *p;
    ssize_t n;
    size_t done;

    p = (const uint8_t *)buffer;
    done = 0u;
    while (done < bytes)
    {
        n = write(fd, &p[done], (bytes - done));
        if (n <= 0)
            return(-1);
        done += (size_t)n;
    }
    return(0);
}

static int32_t SparkDmabufRdmaOpenDevice(
    SparkDmabufRdmaResources *resources,
    const char *device_name,
    int32_t gid_index)
{
    struct ibv_device **devices;
    int32_t count;
    int32_t index;

    devices = ibv_get_device_list(&count);
    if (devices == 0 || count <= 0)
        return(-1);
    for (index=0; index<count; index++)
    {
        if (strcmp(ibv_get_device_name(devices[index]), device_name) == 0)
        {
            resources->verbs_context = ibv_open_device(devices[index]);
            break;
        }
    }
    ibv_free_device_list(devices);
    if (resources->verbs_context == 0)
        return(-2);
    if (ibv_query_gid(resources->verbs_context, 1, gid_index, &resources->gid) != 0)
        return(-3);
    return(0);
}

static int32_t SparkDmabufRdmaInitCudaBuffer(
    SparkDmabufRdmaResources *resources,
    size_t bytes)
{
    CUdevice device;
    CUresult result;

    result = cuInit(0);
    if (result != CUDA_SUCCESS)
        return(-1);
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS)
        return(-2);
    result = cuDevicePrimaryCtxRetain(&resources->cuda_context, device);
    if (result != CUDA_SUCCESS)
        return(-3);
    result = cuCtxSetCurrent(resources->cuda_context);
    if (result != CUDA_SUCCESS)
        return(-4);
    resources->bytes = bytes;
    result = cuMemHostAlloc(
        &resources->host_buffer,
        bytes,
        CU_MEMHOSTALLOC_DEVICEMAP);
    if (result != CUDA_SUCCESS)
        return(-5);
    result = cuMemHostGetDevicePointer(
        &resources->device_buffer,
        resources->host_buffer,
        0);
    if (result != CUDA_SUCCESS)
        return(-6);
    result = cuMemGetHandleForAddressRange(
        &resources->dmabuf_fd,
        resources->device_buffer,
        bytes,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
        0);
    if (result != CUDA_SUCCESS)
        return(-7);
    return(0);
}

static int32_t SparkDmabufRdmaInitVerbs(
    SparkDmabufRdmaResources *resources,
    const char *device_name,
    int32_t gid_index)
{
    struct ibv_qp_init_attr queue_pair_init;
    int32_t access_flags;

    if (SparkDmabufRdmaOpenDevice(resources, device_name, gid_index) < 0)
        return(-1);
    resources->protection_domain = ibv_alloc_pd(resources->verbs_context);
    if (resources->protection_domain == 0)
        return(-2);
    resources->completion_queue = ibv_create_cq(resources->verbs_context, 8, 0, 0, 0);
    if (resources->completion_queue == 0)
        return(-3);
    memset(&queue_pair_init, 0, sizeof(queue_pair_init));
    queue_pair_init.send_cq = resources->completion_queue;
    queue_pair_init.recv_cq = resources->completion_queue;
    queue_pair_init.qp_type = IBV_QPT_RC;
    queue_pair_init.cap.max_send_wr = 8;
    queue_pair_init.cap.max_recv_wr = 1;
    queue_pair_init.cap.max_send_sge = 1;
    queue_pair_init.cap.max_recv_sge = 1;
    resources->queue_pair = ibv_create_qp(resources->protection_domain, &queue_pair_init);
    if (resources->queue_pair == 0)
        return(-4);
    access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
    resources->memory_region = ibv_reg_dmabuf_mr(
        resources->protection_domain,
        0,
        resources->bytes,
        (uintptr_t)resources->host_buffer,
        resources->dmabuf_fd,
        access_flags);
    if (resources->memory_region == 0)
        return(-5);
    resources->psn = (0x123400u + (uint32_t)(getpid() & 0xfffu));
    return(0);
}

static int32_t SparkDmabufRdmaModifyInit(SparkDmabufRdmaResources *resources)
{
    struct ibv_qp_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = (IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    return(ibv_modify_qp(
        resources->queue_pair,
        &attr,
        (IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
        IBV_QP_ACCESS_FLAGS)));
}

static int32_t SparkDmabufRdmaModifyRtr(
    SparkDmabufRdmaResources *resources,
    const SparkDmabufRdmaPeerInfo *peer,
    int32_t gid_index)
{
    struct ibv_qp_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = peer->qpn;
    attr.rq_psn = peer->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = peer->gid;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.port_num = 1;
    return(ibv_modify_qp(
        resources->queue_pair,
        &attr,
        (IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
        IBV_QP_MIN_RNR_TIMER)));
}

static int32_t SparkDmabufRdmaModifyRts(SparkDmabufRdmaResources *resources)
{
    struct ibv_qp_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = resources->psn;
    attr.max_rd_atomic = 1;
    return(ibv_modify_qp(
        resources->queue_pair,
        &attr,
        (IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)));
}

static int32_t SparkDmabufRdmaExchangeInfo(
    int32_t fd,
    SparkDmabufRdmaResources *resources,
    SparkDmabufRdmaPeerInfo *peer)
{
    SparkDmabufRdmaPeerInfo local;

    memset(&local, 0, sizeof(local));
    local.qpn = resources->queue_pair->qp_num;
    local.psn = resources->psn;
    local.rkey = resources->memory_region->rkey;
    local.addr = (uint64_t)(uintptr_t)resources->host_buffer;
    local.gid = resources->gid;
    if (SparkDmabufRdmaWriteFull(fd, &local, sizeof(local)) < 0)
        return(-1);
    if (SparkDmabufRdmaReadFull(fd, peer, sizeof(*peer)) < 0)
        return(-2);
    return(0);
}

static int32_t SparkDmabufRdmaPostWrite(
    SparkDmabufRdmaResources *resources,
    const SparkDmabufRdmaPeerInfo *peer)
{
    struct ibv_send_wr write_request;
    struct ibv_send_wr *bad_write_request;
    struct ibv_sge scatter_gather_entry;
    struct ibv_wc completion;
    int32_t polls;

    memset(&scatter_gather_entry, 0, sizeof(scatter_gather_entry));
    scatter_gather_entry.addr = (uintptr_t)resources->host_buffer;
    scatter_gather_entry.length = (uint32_t)resources->bytes;
    scatter_gather_entry.lkey = resources->memory_region->lkey;
    memset(&write_request, 0, sizeof(write_request));
    write_request.wr_id = 0x1234u;
    write_request.sg_list = &scatter_gather_entry;
    write_request.num_sge = 1;
    write_request.opcode = IBV_WR_RDMA_WRITE;
    write_request.send_flags = IBV_SEND_SIGNALED;
    write_request.wr.rdma.remote_addr = peer->addr;
    write_request.wr.rdma.rkey = peer->rkey;
    if (ibv_post_send(resources->queue_pair, &write_request, &bad_write_request) != 0)
        return(-1);
    for (polls=0; polls<1000000; polls++)
    {
        if (ibv_poll_cq(resources->completion_queue, 1, &completion) == 1)
            return(completion.status == IBV_WC_SUCCESS ? 0 : -2);
    }
    return(-3);
}

static int32_t SparkDmabufRdmaVerifyDevicePattern(
    SparkDmabufRdmaResources *resources,
    uint8_t value)
{
    uint8_t *p;
    CUresult result;
    size_t i;

    p = (uint8_t *)malloc(resources->bytes);
    if (p == 0)
        return(-1);
    result = cuMemcpyDtoH(p, resources->device_buffer, resources->bytes);
    if (result != CUDA_SUCCESS)
    {
        free(p);
        return(-2);
    }
    for (i=0; i<resources->bytes; i++)
    {
        if (p[i] != value)
        {
            free(p);
            return(-3);
        }
    }
    free(p);
    return(0);
}

static void SparkDmabufRdmaCleanup(SparkDmabufRdmaResources *resources)
{
    if (resources->memory_region != 0)
        ibv_dereg_mr(resources->memory_region);
    if (resources->queue_pair != 0)
        ibv_destroy_qp(resources->queue_pair);
    if (resources->completion_queue != 0)
        ibv_destroy_cq(resources->completion_queue);
    if (resources->protection_domain != 0)
        ibv_dealloc_pd(resources->protection_domain);
    if (resources->verbs_context != 0)
        ibv_close_device(resources->verbs_context);
    if (resources->dmabuf_fd >= 0)
        close(resources->dmabuf_fd);
    if (resources->host_buffer != 0)
        cuMemFreeHost(resources->host_buffer);
}

static int32_t SparkDmabufRdmaParseBytes(int32_t argc,char **argv,size_t *bytes_out)
{
    size_t bytes;

    if (bytes_out == 0)
        return(-1);
    bytes = (argc > 6) ? (size_t)strtoull(argv[6], 0, 10) : 1048576u;
    if (bytes == 0u)
        return(-2);
    if ((bytes & 4095u) != 0u)
        bytes = ((bytes + 4095u) & ~(size_t)4095u);
    *bytes_out = bytes;
    return(0);
}

static int32_t SparkDmabufRdmaRunServer(
    SparkDmabufRdmaResources *resources,
    int32_t fd,
    uint8_t pattern)
{
    char done;
    int32_t status;

    if (SparkDmabufRdmaReadFull(fd, &done, 1) < 0)
        return(-1);
    cuCtxSynchronize();
    status = SparkDmabufRdmaVerifyDevicePattern(resources, pattern);
    printf("server_verify=%s bytes=%zu\n",
        status == 0 ? "ok" : "bad",
        resources->bytes);
    return(status == 0 ? 0 : -2);
}

static int32_t SparkDmabufRdmaRunClient(
    SparkDmabufRdmaResources *resources,
    const SparkDmabufRdmaPeerInfo *peer,
    int32_t fd,
    uint8_t pattern)
{
    CUresult result;
    int32_t status;
    char done;

    result = cuMemsetD8(resources->device_buffer, pattern, resources->bytes);
    if (result != CUDA_SUCCESS)
        return(-1);
    cuCtxSynchronize();
    status = SparkDmabufRdmaPostWrite(resources, peer);
    printf("client_rdma_write=%s bytes=%zu\n",
        status == 0 ? "ok" : "bad",
        resources->bytes);
    done = 'D';
    if (SparkDmabufRdmaWriteFull(fd, &done, 1) < 0)
        return(-2);
    return(status == 0 ? 0 : -3);
}

static int32_t SparkDmabufRdmaConnectControl(
    int32_t is_server,
    const char *peer_ip,
    int32_t port)
{
    int32_t listen_fd;
    int32_t fd;

    if (is_server != 0)
    {
        listen_fd = SparkDmabufRdmaTcpListen(port);
        if (listen_fd < 0)
            return(-1);
        fd = accept(listen_fd, 0, 0);
        close(listen_fd);
        return(fd);
    }
    return(SparkDmabufRdmaTcpConnect(peer_ip, port));
}

static int32_t SparkDmabufRdmaRun(int32_t argc,char **argv)
{
    SparkDmabufRdmaResources resources;
    SparkDmabufRdmaPeerInfo peer;
    const char *device_name;
    const char *peer_ip;
    uint8_t pattern;
    int32_t is_server;
    int32_t gid_index;
    int32_t port;
    int32_t fd;
    size_t bytes;
    int32_t status;

    if (argc < 6)
        return(-1);
    memset(&resources, 0, sizeof(resources));
    resources.dmabuf_fd = -1;
    is_server = (strcmp(argv[1], "server") == 0);
    peer_ip = argv[2];
    port = atoi(argv[3]);
    device_name = argv[4];
    gid_index = atoi(argv[5]);
    pattern = (uint8_t)(0x40u + ((uint32_t)port & 0x3fu));
    status = SparkDmabufRdmaParseBytes(argc, argv, &bytes);
    if (status < 0)
        return(-2);
    status = SparkDmabufRdmaInitCudaBuffer(&resources, bytes);
    if (status < 0)
    {
        printf("init_cuda_dmabuf=%d\n", status);
        return(-3);
    }
    status = SparkDmabufRdmaInitVerbs(&resources, device_name, gid_index);
    if (status < 0)
    {
        printf("init_verbs=%d errno=%d %s\n", status, errno, strerror(errno));
        SparkDmabufRdmaCleanup(&resources);
        return(-4);
    }
    if (SparkDmabufRdmaModifyInit(&resources) != 0)
    {
        printf("modify_init failed errno=%d %s\n", errno, strerror(errno));
        SparkDmabufRdmaCleanup(&resources);
        return(-5);
    }
    fd = SparkDmabufRdmaConnectControl(is_server, peer_ip, port);
    if (fd < 0)
    {
        printf("tcp failed %d errno=%d %s\n", fd, errno, strerror(errno));
        SparkDmabufRdmaCleanup(&resources);
        return(-6);
    }
    status = SparkDmabufRdmaExchangeInfo(fd, &resources, &peer);
    if (status < 0)
    {
        printf("exchange failed\n");
        close(fd);
        SparkDmabufRdmaCleanup(&resources);
        return(-7);
    }
    if (SparkDmabufRdmaModifyRtr(&resources, &peer, gid_index) != 0 ||
        SparkDmabufRdmaModifyRts(&resources) != 0)
    {
        printf("qp transition failed errno=%d %s\n", errno, strerror(errno));
        close(fd);
        SparkDmabufRdmaCleanup(&resources);
        return(-8);
    }
    status = is_server != 0 ?
        SparkDmabufRdmaRunServer(&resources, fd, pattern) :
        SparkDmabufRdmaRunClient(&resources, &peer, fd, pattern);
    close(fd);
    SparkDmabufRdmaCleanup(&resources);
    return(status);
}

int main(int argc,char **argv)
{
    int32_t status;

    if (argc < 6)
    {
        fprintf(stderr,
            "usage: %s server|client <peer-ip|listen-ip> <port> <ibdev> <gid-index> [bytes]\n",
            argv[0]);
        return(2);
    }
    status = SparkDmabufRdmaRun(argc, argv);
    return(status == 0 ? 0 : 1);
}

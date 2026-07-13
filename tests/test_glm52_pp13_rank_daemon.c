#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define main SparkTestGlm52Pp13RankDaemonMain
#include "../tools/sparkpipe_glm52_pp13_rank_daemon.c"
#undef main

static void SparkTestRankDaemonReadsSplitResidentMessage(void)
{
    static SparkGlm52Pp13DaemonRuntime runtime;
    SparkGlm52CudaResidentIpcHeader header;
    SparkGlm52CudaResidentIpcHeader received_header;
    SparkGlm52CudaResidentIpcSubmitResult payload;
    int32_t sockets[2];
    uint32_t header_split;
    uint32_t payload_split;

    memset(&runtime,0,sizeof(runtime));
    memset(&payload,0,sizeof(payload));
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets) == 0);
    runtime.cuda_resident_fd = sockets[0];
    assert(SparkGlm52Pp13DaemonSetNonblocking(sockets[0]) == 0);
    payload.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES;
    payload.status = SPARK_STATUS_OK;
    SparkGlm52CudaResidentIpcInitializeHeader(
        &header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT,
        4u,9u,sizeof(payload));
    header_split = (uint32_t)sizeof(header) / 2u;
    payload_split = (uint32_t)sizeof(payload) / 2u;
    assert(write(sockets[1],&header,header_split) == (ssize_t)header_split);
    assert(SparkGlm52Pp13DaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_BUSY);
    assert(runtime.cuda_resident_read_header_offset == header_split);
    assert(write(sockets[1],((const uint8_t *)&header) + header_split,
        sizeof(header) - header_split) ==
        (ssize_t)(sizeof(header) - header_split));
    assert(write(sockets[1],&payload,payload_split) ==
        (ssize_t)payload_split);
    assert(SparkGlm52Pp13DaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_BUSY);
    assert(runtime.cuda_resident_read_header_offset == sizeof(header));
    assert(runtime.cuda_resident_read_payload_offset == payload_split);
    assert(write(sockets[1],((const uint8_t *)&payload) + payload_split,
        sizeof(payload) - payload_split) ==
        (ssize_t)(sizeof(payload) - payload_split));
    assert(SparkGlm52Pp13DaemonReadResidentMessage(
        &runtime,0u,&received_header) == SPARK_STATUS_OK);
    assert(received_header.kind ==
        SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT);
    assert(memcmp(runtime.cuda_resident_payload,&payload,sizeof(payload)) == 0);
    assert(runtime.cuda_resident_read_header_offset == 0u);
    assert(runtime.cuda_resident_read_payload_offset == 0u);
    close(sockets[0]);
    close(sockets[1]);
}

int main(void)
{
    SparkTestRankDaemonReadsSplitResidentMessage();
    return 0;
}

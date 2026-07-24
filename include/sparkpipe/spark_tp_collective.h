#ifndef SPARKPIPE_SPARK_TP_COLLECTIVE_H
#define SPARKPIPE_SPARK_TP_COLLECTIVE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

// Tensor-parallel collective: the all-reduce that closes every attention and
// MLP block under TP. This is the reference implementation over TCP sockets,
// correct and deterministic, standing in for the fabric implementation the
// direct-connect switches enable; the interface is the contract and the
// fabric version drops in behind it. On GB10 the production path is the
// unified-memory flag-polling design: the GPU writes its partial to a
// unified buffer, the transport exchanges with peers, and the kernel
// spin-polls a sequence flag and reduces locally, with no copies and no
// relaunch, which is what makes the two microsecond per step budget
// plausible; none of that changes this interface.
//
// The algorithm is recursive doubling: log2(degree) steps, step s exchanging
// with the partner whose rank differs in bit s, both sides accumulating
// local plus remote. Because IEEE addition of non-NaN operands is bitwise
// commutative, every rank holds the bitwise-identical sum after the final
// step. That determinism is a hard requirement, not a nicety: the reduced
// hidden state feeds the router, and ranks that disagree in even one
// mantissa bit can route to different experts and diverge unrecoverably.
//
// Connection establishment is deadlock-free by ordering: every rank listens
// on its own port, connects to lower-ranked partners, and accepts from
// higher-ranked ones, with a rank-identifying handshake mapping accepted
// sockets to steps. Within a step the lower rank sends before receiving and
// the higher receives before sending, so arbitrarily large buffers cannot
// deadlock on socket backpressure. Degrees one, two, four, eight, and
// sixteen are accepted; degree one is a no-op so single-rank shapes need no
// special casing.

#define SPARK_TP_COLLECTIVE_ABI_VERSION 1u
#define SPARK_TP_COLLECTIVE_MAX_STEPS 4u
#define SPARK_TP_COLLECTIVE_HOST_NAME_BYTES 64u

typedef struct SparkTpCollectivePeer
{
	char host_name[SPARK_TP_COLLECTIVE_HOST_NAME_BYTES];
	uint16_t port;
	uint16_t reserved0;
	uint32_t reserved1;
} SparkTpCollectivePeer;

typedef struct SparkTpCollectiveConfig
{
	uint32_t abi_version;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint16_t listen_port;
	uint16_t reserved0;
	uint32_t connect_timeout_milli;
	// Step partner peers: peers[s] is the rank whose index differs in bit s,
	// so exactly log2(tp_degree) entries are read.
	SparkTpCollectivePeer peers[SPARK_TP_COLLECTIVE_MAX_STEPS];
} SparkTpCollectiveConfig;

typedef struct SparkTpCollective
{
	uint32_t abi_version;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t step_count;
	int32_t listen_socket;
	int32_t step_sockets[SPARK_TP_COLLECTIVE_MAX_STEPS];
} SparkTpCollective;

// Bind, connect to lower partners, accept from higher partners. Blocks until
// the full step-socket set exists or the timeout elapses; a partial group
// fails closed and leaves no sockets open.
SparkStatus SparkTpCollectiveCreate(
	const SparkTpCollectiveConfig *config,
	SparkTpCollective *collective_out);

// In-place sum all-reduce of element_count floats. scratch must hold
// element_count floats and is caller owned. After success every rank of the
// group holds the bitwise-identical elementwise sum. Degree one returns
// immediately.
SparkStatus SparkTpCollectiveAllReduceSumF32(
	SparkTpCollective *collective,
	float *values,
	uint64_t element_count,
	float *scratch);

void SparkTpCollectiveDestroy(SparkTpCollective *collective);

#endif

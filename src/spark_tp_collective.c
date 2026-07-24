#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_collective.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static uint64_t SparkTpCollectiveNowMilli(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static SparkStatus SparkTpCollectiveSendAll(int32_t socket_descriptor,const void *data,uint64_t data_bytes)
{
	const uint8_t *bytes = (const uint8_t *)data;
	uint64_t sent = 0u;
	while (sent < data_bytes)
	{
		ssize_t result = send(socket_descriptor,bytes + sent,(size_t)(data_bytes - sent),0);
		if (result <= 0)
		{
			if (result < 0 && errno == EINTR)
				continue;
			return SPARK_STATUS_IO_ERROR;
		}
		sent += (uint64_t)result;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveReceiveAll(int32_t socket_descriptor,void *data,uint64_t data_bytes)
{
	uint8_t *bytes = (uint8_t *)data;
	uint64_t received = 0u;
	while (received < data_bytes)
	{
		ssize_t result = recv(socket_descriptor,bytes + received,(size_t)(data_bytes - received),0);
		if (result <= 0)
		{
			if (result < 0 && errno == EINTR)
				continue;
			return SPARK_STATUS_IO_ERROR;
		}
		received += (uint64_t)result;
	}
	return SPARK_STATUS_OK;
}

static void SparkTpCollectiveCloseAll(SparkTpCollective *collective)
{
	uint32_t step_index;
	if (collective->listen_socket >= 0)
		close(collective->listen_socket);
	collective->listen_socket = -1;
	for (step_index = 0u; step_index < SPARK_TP_COLLECTIVE_MAX_STEPS; ++step_index)
	{
		if (collective->step_sockets[step_index] >= 0)
			close(collective->step_sockets[step_index]);
		collective->step_sockets[step_index] = -1;
	}
}

static SparkStatus SparkTpCollectiveListen(SparkTpCollective *collective,uint16_t listen_port)
{
	struct sockaddr_in address;
	int reuse = 1;
	collective->listen_socket = socket(AF_INET,SOCK_STREAM,0);
	if (collective->listen_socket < 0)
		return SPARK_STATUS_IO_ERROR;
	setsockopt(collective->listen_socket,SOL_SOCKET,SO_REUSEADDR,&reuse,sizeof(reuse));
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons(listen_port);
	if (bind(collective->listen_socket,(struct sockaddr *)&address,sizeof(address)) != 0 ||
		listen(collective->listen_socket,SPARK_TP_COLLECTIVE_MAX_STEPS) != 0)
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTpCollectiveConnectPeer(const SparkTpCollectivePeer *peer,uint32_t timeout_milli,uint32_t self_rank,int32_t *socket_out)
{
	struct sockaddr_in address;
	uint64_t deadline = SparkTpCollectiveNowMilli() + timeout_milli;
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons(peer->port);
	if (inet_pton(AF_INET,peer->host_name,&address.sin_addr) != 1)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (;;)
	{
		int32_t connect_socket = socket(AF_INET,SOCK_STREAM,0);
		int nodelay = 1;
		if (connect_socket < 0)
			return SPARK_STATUS_IO_ERROR;
		setsockopt(connect_socket,IPPROTO_TCP,TCP_NODELAY,&nodelay,sizeof(nodelay));
		if (connect(connect_socket,(struct sockaddr *)&address,sizeof(address)) == 0)
		{
			uint32_t rank_wire = self_rank;
			if (SparkTpCollectiveSendAll(connect_socket,&rank_wire,sizeof(rank_wire)) != SPARK_STATUS_OK)
			{
				close(connect_socket);
				return SPARK_STATUS_IO_ERROR;
			}
			*socket_out = connect_socket;
			return SPARK_STATUS_OK;
		}
		close(connect_socket);
		if (SparkTpCollectiveNowMilli() >= deadline)
			return SPARK_STATUS_IO_ERROR;
		{
			struct timespec pause = {0,2000000};
			nanosleep(&pause,0);
		}
	}
}

static SparkStatus SparkTpCollectiveAcceptPeers(SparkTpCollective *collective,uint32_t expected_count)
{
	uint32_t accepted = 0u;
	while (accepted < expected_count)
	{
		uint32_t remote_rank,step_index;
		int nodelay = 1;
		int32_t accept_socket = accept(collective->listen_socket,0,0);
		if (accept_socket < 0)
		{
			if (errno == EINTR)
				continue;
			if (errno == EAGAIN || errno == EWOULDBLOCK)
				return SPARK_STATUS_IO_ERROR;
			return SPARK_STATUS_IO_ERROR;
		}
		setsockopt(accept_socket,IPPROTO_TCP,TCP_NODELAY,&nodelay,sizeof(nodelay));
		if (SparkTpCollectiveReceiveAll(accept_socket,&remote_rank,sizeof(remote_rank)) != SPARK_STATUS_OK ||
			(remote_rank ^ collective->tp_rank) == 0u)
		{
			close(accept_socket);
			return SPARK_STATUS_IO_ERROR;
		}
		// The partner's rank differs from ours in exactly one bit; that bit is
		// the step index. Anything else is a wiring error and fails closed.
		{
			uint32_t difference = remote_rank ^ collective->tp_rank;
			if ((difference & (difference - 1u)) != 0u)
			{
				close(accept_socket);
				return SPARK_STATUS_VALIDATION_FAILED;
			}
			step_index = 0u;
			while ((difference >> (step_index + 1u)) != 0u)
				step_index += 1u;
		}
		if (step_index >= collective->step_count ||
			collective->step_sockets[step_index] >= 0)
		{
			close(accept_socket);
			return SPARK_STATUS_VALIDATION_FAILED;
		}
		collective->step_sockets[step_index] = accept_socket;
		accepted += 1u;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpCollectiveCreate(
	const SparkTpCollectiveConfig *config,
	SparkTpCollective *collective_out)
{
	uint32_t step_index,connect_count,accept_count;
	SparkStatus status;
	if (config == 0 || collective_out == 0 ||
		config->abi_version != SPARK_TP_COLLECTIVE_ABI_VERSION)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (config->tp_degree != 1u && config->tp_degree != 2u &&
		config->tp_degree != 4u && config->tp_degree != 8u &&
		config->tp_degree != 16u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (config->tp_rank >= config->tp_degree)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(collective_out,0,sizeof(*collective_out));
	collective_out->abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
	collective_out->tp_degree = config->tp_degree;
	collective_out->tp_rank = config->tp_rank;
	collective_out->listen_socket = -1;
	for (step_index = 0u; step_index < SPARK_TP_COLLECTIVE_MAX_STEPS; ++step_index)
		collective_out->step_sockets[step_index] = -1;
	collective_out->step_count = 0u;
	while ((config->tp_degree >> (collective_out->step_count + 1u)) != 0u)
		collective_out->step_count += 1u;
	if (config->tp_degree == 1u)
		return SPARK_STATUS_OK;
	status = SparkTpCollectiveListen(collective_out,config->listen_port);
	if (status == SPARK_STATUS_OK && config->connect_timeout_milli != 0u)
	{
		struct timeval accept_timeout;
		accept_timeout.tv_sec = config->connect_timeout_milli / 1000u;
		accept_timeout.tv_usec = (config->connect_timeout_milli % 1000u) * 1000u;
		setsockopt(collective_out->listen_socket,SOL_SOCKET,SO_RCVTIMEO,&accept_timeout,sizeof(accept_timeout));
	}
	connect_count = 0u;
	accept_count = 0u;
	for (step_index = 0u; status == SPARK_STATUS_OK && step_index < collective_out->step_count; ++step_index)
	{
		uint32_t partner = config->tp_rank ^ (1u << step_index);
		if (partner < config->tp_rank)
		{
			status = SparkTpCollectiveConnectPeer(&config->peers[step_index],config->connect_timeout_milli,config->tp_rank,&collective_out->step_sockets[step_index]);
			connect_count += 1u;
		}
		else
			accept_count += 1u;
	}
	if (status == SPARK_STATUS_OK)
		status = SparkTpCollectiveAcceptPeers(collective_out,accept_count);
	if (status == SPARK_STATUS_OK && connect_count + accept_count != collective_out->step_count)
		status = SPARK_STATUS_VALIDATION_FAILED;
	if (status != SPARK_STATUS_OK)
		SparkTpCollectiveCloseAll(collective_out);
	return status;
}

SparkStatus SparkTpCollectiveAllReduceSumF32(
	SparkTpCollective *collective,
	float *values,
	uint64_t element_count,
	float *scratch)
{
	uint32_t step_index;
	uint64_t element_index,buffer_bytes;
	SparkStatus status;
	if (collective == 0 || values == 0 || element_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (collective->tp_degree == 1u)
		return SPARK_STATUS_OK;
	if (scratch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	buffer_bytes = element_count * sizeof(float);
	for (step_index = 0u; step_index < collective->step_count; ++step_index)
	{
		uint32_t partner = collective->tp_rank ^ (1u << step_index);
		int32_t step_socket = collective->step_sockets[step_index];
		if (step_socket < 0)
			return SPARK_STATUS_INVALID_ARGUMENT;
		// Lower rank sends first, higher receives first: deadlock free at any
		// buffer size, and both sides then compute local plus remote, which is
		// bitwise identical for non-NaN IEEE operands.
		if (collective->tp_rank < partner)
		{
			status = SparkTpCollectiveSendAll(step_socket,values,buffer_bytes);
			if (status == SPARK_STATUS_OK)
				status = SparkTpCollectiveReceiveAll(step_socket,scratch,buffer_bytes);
		}
		else
		{
			status = SparkTpCollectiveReceiveAll(step_socket,scratch,buffer_bytes);
			if (status == SPARK_STATUS_OK)
				status = SparkTpCollectiveSendAll(step_socket,values,buffer_bytes);
		}
		if (status != SPARK_STATUS_OK)
			return status;
		for (element_index = 0u; element_index < element_count; ++element_index)
			values[element_index] = values[element_index] + scratch[element_index];
	}
	return SPARK_STATUS_OK;
}

void SparkTpCollectiveDestroy(SparkTpCollective *collective)
{
	if (collective == 0)
		return;
	SparkTpCollectiveCloseAll(collective);
}

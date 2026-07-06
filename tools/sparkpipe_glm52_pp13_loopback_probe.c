#define _POSIX_C_SOURCE 200112L

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_pp13_runtime.h"

#define SPARK_GLM52_PP13_LOOPBACK_MAGIC 0x314b424cu
#define SPARK_GLM52_PP13_LOOPBACK_VERSION 2u
#define SPARK_GLM52_PP13_LOOPBACK_MAX_HOPS 16u
#define SPARK_GLM52_PP13_LOOPBACK_DEFAULT_PORT_BASE 54100u
#define SPARK_GLM52_PP13_LOOPBACK_DEFAULT_TIMEOUT_MS 30000u
#define SPARK_GLM52_PP13_LOOPBACK_CONNECT_RETRY_NS 250000ull

typedef struct SparkGlm52Pp13LoopbackPacket
{
	uint32_t magic;
	uint32_t version;
	uint32_t descriptor_bytes;
	uint32_t origin_rank;
	uint32_t sequence;
	uint32_t hop_count;
	uint32_t max_hops;
	uint32_t reserved;
	uint64_t created_wall_ns;
	uint64_t completed_wall_ns;
	uint64_t created_mono_ns;
	uint64_t completed_mono_ns;
	uint32_t ranks[SPARK_GLM52_PP13_LOOPBACK_MAX_HOPS];
	uint64_t wall_ns[SPARK_GLM52_PP13_LOOPBACK_MAX_HOPS];
} SparkGlm52Pp13LoopbackPacket;

typedef struct SparkGlm52Pp13LoopbackConfig
{
	uint32_t rank_index;
	uint32_t origin_rank;
	uint32_t launch_origin;
	uint32_t port_base;
	uint32_t timeout_ms;
	uint32_t sequence;
	uint32_t repeat_count;
} SparkGlm52Pp13LoopbackConfig;

typedef struct SparkGlm52Pp13LoopbackRuntime
{
	SparkGlm52Pp13LoopbackConfig config;
	int32_t listen_fd;
	int32_t input_fd;
	int32_t output_fd;
	uint32_t input_offset;
	uint32_t output_offset;
	uint32_t output_active;
	uint32_t output_connecting;
	uint32_t launched;
	uint32_t completed_count;
	uint64_t next_connect_mono_ns;
	SparkGlm52Pp13LoopbackPacket input_packet;
	SparkGlm52Pp13LoopbackPacket output_packet;
} SparkGlm52Pp13LoopbackRuntime;

static uint64_t SparkGlm52Pp13LoopbackWallNs(void)
{
	struct timespec ts;
	if (clock_gettime(CLOCK_REALTIME,&ts) != 0)
		return 0u;
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static uint64_t SparkGlm52Pp13LoopbackMonoNs(void)
{
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC,&ts) != 0)
		return 0u;
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static int32_t SparkGlm52Pp13LoopbackParseU32(const char *text,uint32_t *value_out)
{
	uint64_t value;
	uint32_t index;
	if (text == 0 || text[0] == '\0' || value_out == 0)
		return -1;
	value = 0u;
	for (index=0u; text[index] != '\0'; index++)
	{
		if (text[index] < '0' || text[index] > '9')
			return -2;
		value = ((value * 10u) + (uint32_t)(text[index] - '0'));
		if (value > 0xffffffffull)
			return -3;
	}
	*value_out = (uint32_t)value;
	return 0;
}

static int32_t SparkGlm52Pp13LoopbackSetNonblocking(int32_t fd)
{
	int32_t flags;
	flags = fcntl(fd,F_GETFL,0);
	if (flags < 0)
		return -1;
	return fcntl(fd,F_SETFL,(flags | O_NONBLOCK)) < 0 ? -2 : 0;
}

static int32_t SparkGlm52Pp13LoopbackConfigureSocket(int32_t fd)
{
	int32_t option;
	if (fd < 0)
		return -1;
	option = 1;
	(void)setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&option,sizeof(option));
	return SparkGlm52Pp13LoopbackSetNonblocking(fd);
}

static void SparkGlm52Pp13LoopbackInitConfig(SparkGlm52Pp13LoopbackConfig *config)
{
	memset(config,0,sizeof(*config));
	config->origin_rank = 0u;
	config->port_base = SPARK_GLM52_PP13_LOOPBACK_DEFAULT_PORT_BASE;
	config->timeout_ms = SPARK_GLM52_PP13_LOOPBACK_DEFAULT_TIMEOUT_MS;
	config->sequence = 1u;
	config->repeat_count = 1u;
}

static int32_t SparkGlm52Pp13LoopbackApplyArg(
	SparkGlm52Pp13LoopbackConfig *config,
	int32_t argc,
	char **argv,
	int32_t *index)
{
	if (strcmp(argv[*index],"--rank") == 0 && (*index + 1) < argc)
	{
		*index += 1;
		return SparkGlm52Pp13LoopbackParseU32(argv[*index],&config->rank_index);
	}
	if (strcmp(argv[*index],"--origin-rank") == 0 && (*index + 1) < argc)
	{
		*index += 1;
		return SparkGlm52Pp13LoopbackParseU32(argv[*index],&config->origin_rank);
	}
	if (strcmp(argv[*index],"--port-base") == 0 && (*index + 1) < argc)
	{
		*index += 1;
		return SparkGlm52Pp13LoopbackParseU32(argv[*index],&config->port_base);
	}
	if (strcmp(argv[*index],"--timeout-ms") == 0 && (*index + 1) < argc)
	{
		*index += 1;
		return SparkGlm52Pp13LoopbackParseU32(argv[*index],&config->timeout_ms);
	}
	if (strcmp(argv[*index],"--sequence") == 0 && (*index + 1) < argc)
	{
		*index += 1;
		return SparkGlm52Pp13LoopbackParseU32(argv[*index],&config->sequence);
	}
	if (strcmp(argv[*index],"--repeat-count") == 0 && (*index + 1) < argc)
	{
		*index += 1;
		return SparkGlm52Pp13LoopbackParseU32(argv[*index],&config->repeat_count);
	}
	if (strcmp(argv[*index],"--launch-origin") == 0)
	{
		config->launch_origin = 1u;
		return 0;
	}
	return -1;
}

static int32_t SparkGlm52Pp13LoopbackParseArgs(
	SparkGlm52Pp13LoopbackConfig *config,
	int32_t argc,
	char **argv)
{
	int32_t index;
	SparkGlm52Pp13LoopbackInitConfig(config);
	for (index=1; index<argc; index++)
	{
		if (SparkGlm52Pp13LoopbackApplyArg(config,argc,argv,&index) < 0)
			return -1;
	}
	if (config->rank_index >= SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT ||
		config->origin_rank >= SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT ||
		config->repeat_count == 0u ||
		config->port_base > (65535u - SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT))
		return -2;
	return 0;
}

static int32_t SparkGlm52Pp13LoopbackListen(uint32_t port)
{
	struct sockaddr_in address;
	int32_t fd;
	int32_t option;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if (fd < 0)
		return -1;
	option = 1;
	setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&option,sizeof(option));
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons((uint16_t)port);
	if (bind(fd,(struct sockaddr *)&address,sizeof(address)) < 0 ||
		listen(fd,16) < 0 ||
		SparkGlm52Pp13LoopbackConfigureSocket(fd) < 0)
	{
		close(fd);
		return -2;
	}
	return fd;
}

static int32_t SparkGlm52Pp13LoopbackStartConnect(
	const char *host,
	uint32_t port,
	uint32_t *connecting_out)
{
	struct addrinfo hints;
	struct addrinfo *results;
	struct addrinfo *entry;
	char service[16];
	int32_t fd;
	int32_t connect_status;
	if (connecting_out == 0)
		return -1;
	*connecting_out = 0u;
	memset(&hints,0,sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	snprintf(service,sizeof(service),"%u",port);
	if (getaddrinfo(host,service,&hints,&results) != 0)
		return -2;
	fd = -1;
	for (entry=results; entry!=0; entry=entry->ai_next)
	{
		fd = socket(entry->ai_family,entry->ai_socktype,entry->ai_protocol);
		if (fd < 0)
			continue;
		if (SparkGlm52Pp13LoopbackConfigureSocket(fd) < 0)
		{
			close(fd);
			fd = -1;
			continue;
		}
		connect_status = connect(fd,entry->ai_addr,entry->ai_addrlen);
		if (connect_status == 0)
			break;
		if (errno == EINPROGRESS)
		{
			*connecting_out = 1u;
			break;
		}
		close(fd);
		fd = -1;
	}
	freeaddrinfo(results);
	return fd;
}

static int32_t SparkGlm52Pp13LoopbackFinishConnect(
	SparkGlm52Pp13LoopbackRuntime *rt)
{
	socklen_t error_bytes;
	int32_t error_value;
	if (rt->output_fd < 0 || rt->output_connecting == 0u)
		return 0;
	error_value = 0;
	error_bytes = (socklen_t)sizeof(error_value);
	if (getsockopt(rt->output_fd,SOL_SOCKET,SO_ERROR,&error_value,&error_bytes) < 0)
	{
		close(rt->output_fd);
		rt->output_fd = -1;
		rt->output_connecting = 0u;
		return 1;
	}
	if (error_value == 0)
	{
		rt->output_connecting = 0u;
		rt->next_connect_mono_ns = 0u;
		return 1;
	}
	close(rt->output_fd);
	rt->output_fd = -1;
	rt->output_connecting = 0u;
	rt->next_connect_mono_ns = SparkGlm52Pp13LoopbackMonoNs() +
		SPARK_GLM52_PP13_LOOPBACK_CONNECT_RETRY_NS;
	return 1;
}

static uint32_t SparkGlm52Pp13LoopbackAcceptInput(SparkGlm52Pp13LoopbackRuntime *rt)
{
	int32_t fd;
	if (rt->input_fd >= 0)
		return 0u;
	fd = accept(rt->listen_fd,0,0);
	if (fd < 0)
		return 0u;
	if (SparkGlm52Pp13LoopbackConfigureSocket(fd) < 0)
	{
		close(fd);
		return 0u;
	}
	rt->input_fd = fd;
	return 1u;
}

static uint32_t SparkGlm52Pp13LoopbackConnectOutput(SparkGlm52Pp13LoopbackRuntime *rt)
{
	char host[SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES];
	uint64_t now_ns;
	uint32_t next_rank;
	uint32_t connecting;
	if (rt->output_fd >= 0)
		return 0u;
	if (rt->output_active == 0u && rt->launched != 0u)
		return 0u;
	now_ns = SparkGlm52Pp13LoopbackMonoNs();
	if (rt->next_connect_mono_ns != 0u &&
		now_ns < rt->next_connect_mono_ns)
		return 0u;
	next_rank = (rt->config.rank_index + 1u) %
		SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT;
	if (SparkGlm52Pp13RuntimeRankHostName(next_rank,host,sizeof(host)) !=
		SPARK_STATUS_OK)
		return 0u;
	connecting = 0u;
	rt->output_fd = SparkGlm52Pp13LoopbackStartConnect(
		host,
		rt->config.port_base + next_rank,
		&connecting);
	rt->output_connecting = connecting;
	rt->next_connect_mono_ns = rt->output_fd >= 0 ? 0u :
		now_ns + SPARK_GLM52_PP13_LOOPBACK_CONNECT_RETRY_NS;
	return rt->output_fd >= 0 ? 1u : 0u;
}

static uint32_t SparkGlm52Pp13LoopbackStartOrigin(SparkGlm52Pp13LoopbackRuntime *rt)
{
	if (rt->config.launch_origin == 0u || rt->launched != 0u ||
		rt->completed_count >= rt->config.repeat_count ||
		rt->config.rank_index != rt->config.origin_rank)
		return 0u;
	memset(&rt->output_packet,0,sizeof(rt->output_packet));
	rt->output_packet.magic = SPARK_GLM52_PP13_LOOPBACK_MAGIC;
	rt->output_packet.version = SPARK_GLM52_PP13_LOOPBACK_VERSION;
	rt->output_packet.descriptor_bytes = (uint32_t)sizeof(rt->output_packet);
	rt->output_packet.origin_rank = rt->config.origin_rank;
	rt->output_packet.sequence = rt->config.sequence + rt->completed_count;
	rt->output_packet.max_hops = SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT;
	rt->output_packet.created_wall_ns = SparkGlm52Pp13LoopbackWallNs();
	rt->output_packet.created_mono_ns = SparkGlm52Pp13LoopbackMonoNs();
	rt->output_offset = 0u;
	rt->output_active = 1u;
	rt->launched = 1u;
	printf("loopback_launch rank=%u sequence=%u wall_ns=%llu\n",
		rt->config.rank_index,
		rt->config.sequence,
		(unsigned long long)rt->output_packet.created_wall_ns);
	fflush(stdout);
	return 1u;
}

static int32_t SparkGlm52Pp13LoopbackReadInput(SparkGlm52Pp13LoopbackRuntime *rt)
{
	ssize_t got;
	uint32_t remaining;
	if (rt->input_fd < 0)
		return 0;
	remaining = ((uint32_t)sizeof(rt->input_packet) - rt->input_offset);
	got = read(rt->input_fd,((uint8_t *)&rt->input_packet) + rt->input_offset,remaining);
	if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
		return 0;
	if (got <= 0)
	{
		close(rt->input_fd);
		rt->input_fd = -1;
		rt->input_offset = 0u;
		return 0;
	}
	rt->input_offset += (uint32_t)got;
	return rt->input_offset == sizeof(rt->input_packet) ? 1 : 0;
}

static int32_t SparkGlm52Pp13LoopbackQueueOutput(
	SparkGlm52Pp13LoopbackRuntime *rt)
{
	uint32_t index;
	if (rt->output_active != 0u ||
		rt->input_packet.magic != SPARK_GLM52_PP13_LOOPBACK_MAGIC ||
		rt->input_packet.version != SPARK_GLM52_PP13_LOOPBACK_VERSION ||
		rt->input_packet.descriptor_bytes != sizeof(rt->input_packet) ||
		rt->input_packet.hop_count >= rt->input_packet.max_hops ||
		rt->input_packet.max_hops > SPARK_GLM52_PP13_LOOPBACK_MAX_HOPS)
		return -1;
	index = rt->input_packet.hop_count;
	rt->input_packet.ranks[index] = rt->config.rank_index;
	rt->input_packet.wall_ns[index] = SparkGlm52Pp13LoopbackWallNs();
	rt->input_packet.hop_count += 1u;
	printf("loopback_hop rank=%u sequence=%u hop=%u wall_ns=%llu\n",
		rt->config.rank_index,
		rt->input_packet.sequence,
		rt->input_packet.hop_count,
		(unsigned long long)rt->input_packet.wall_ns[index]);
	fflush(stdout);
	if (rt->config.rank_index == rt->input_packet.origin_rank && index != 0u)
		return 1;
	rt->output_packet = rt->input_packet;
	rt->output_offset = 0u;
	rt->output_active = 1u;
	return 0;
}

static int32_t SparkGlm52Pp13LoopbackWriteOutput(SparkGlm52Pp13LoopbackRuntime *rt)
{
	ssize_t wrote;
	uint32_t remaining;
	if (rt->output_fd < 0 || rt->output_active == 0u)
		return 0;
	if (rt->output_connecting != 0u)
		return 0;
	remaining = ((uint32_t)sizeof(rt->output_packet) - rt->output_offset);
	wrote = write(rt->output_fd,((uint8_t *)&rt->output_packet) + rt->output_offset,remaining);
	if (wrote < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
		return 0;
	if (wrote <= 0)
	{
		close(rt->output_fd);
		rt->output_fd = -1;
		rt->output_connecting = 0u;
		rt->output_offset = 0u;
		rt->next_connect_mono_ns = SparkGlm52Pp13LoopbackMonoNs() +
			SPARK_GLM52_PP13_LOOPBACK_CONNECT_RETRY_NS;
		return 1;
	}
	rt->output_offset += (uint32_t)wrote;
	if (rt->output_offset < sizeof(rt->output_packet))
		return 1;
	printf("loopback_forward rank=%u sequence=%u hop=%u\n",
		rt->config.rank_index,
		rt->output_packet.sequence,
		rt->output_packet.hop_count);
	fflush(stdout);
	rt->output_active = 0u;
	rt->output_offset = 0u;
	return 1;
}

static void SparkGlm52Pp13LoopbackAddSelectFd(
	int32_t fd,
	fd_set *set,
	int32_t *max_fd)
{
	if (fd < 0 || set == 0 || max_fd == 0)
		return;
	FD_SET(fd,set);
	if (fd > *max_fd)
		*max_fd = fd;
}

static void SparkGlm52Pp13LoopbackWaitForEvents(
	SparkGlm52Pp13LoopbackRuntime *rt,
	uint64_t timeout_ns)
{
	struct timeval timeout;
	fd_set read_fds;
	fd_set write_fds;
	int32_t max_fd;
	FD_ZERO(&read_fds);
	FD_ZERO(&write_fds);
	max_fd = -1;
	if (rt->input_fd < 0)
		SparkGlm52Pp13LoopbackAddSelectFd(rt->listen_fd,&read_fds,&max_fd);
	if (rt->input_fd >= 0)
		SparkGlm52Pp13LoopbackAddSelectFd(rt->input_fd,&read_fds,&max_fd);
	if ((rt->output_active != 0u || rt->output_connecting != 0u) &&
		rt->output_fd >= 0)
		SparkGlm52Pp13LoopbackAddSelectFd(rt->output_fd,&write_fds,&max_fd);
	if (max_fd < 0)
		return;
	timeout.tv_sec = (time_t)(timeout_ns / 1000000000ull);
	timeout.tv_usec = (suseconds_t)((timeout_ns % 1000000000ull) / 1000ull);
	if (timeout_ns != 0u && timeout.tv_sec == 0 && timeout.tv_usec == 0)
		timeout.tv_usec = 1;
	for (;;)
	{
		if (select(max_fd + 1,&read_fds,&write_fds,0,&timeout) < 0 &&
			errno == EINTR)
			return;
		return;
	}
}

static void SparkGlm52Pp13LoopbackPrintComplete(
	const SparkGlm52Pp13LoopbackPacket *packet)
{
	uint64_t elapsed_ns;
	uint64_t average_hop_ns;
	uint32_t index;
	elapsed_ns = packet->completed_mono_ns - packet->created_mono_ns;
	average_hop_ns = packet->hop_count != 0u ?
		elapsed_ns / (uint64_t)packet->hop_count : 0u;
	printf("loopback_complete sequence=%u origin=%u hops=%u created_wall_ns=%llu completed_wall_ns=%llu created_mono_ns=%llu completed_mono_ns=%llu elapsed_ns=%llu average_hop_ns=%llu\n",
		packet->sequence,
		packet->origin_rank,
		packet->hop_count,
		(unsigned long long)packet->created_wall_ns,
		(unsigned long long)packet->completed_wall_ns,
		(unsigned long long)packet->created_mono_ns,
		(unsigned long long)packet->completed_mono_ns,
		(unsigned long long)elapsed_ns,
		(unsigned long long)average_hop_ns);
	for (index=0u; index<packet->hop_count; index++)
		printf("loopback_trace index=%u rank=%u wall_ns=%llu\n",
			index,
			packet->ranks[index],
			(unsigned long long)packet->wall_ns[index]);
	fflush(stdout);
}

static int32_t SparkGlm52Pp13LoopbackRun(SparkGlm52Pp13LoopbackRuntime *rt)
{
	uint64_t start_ns;
	uint64_t now_ns;
	uint64_t deadline_ns;
	uint64_t wait_ns;
	int32_t read_status;
	int32_t queue_status;
	int32_t write_status;
	uint32_t progress;
	start_ns = SparkGlm52Pp13LoopbackMonoNs();
	deadline_ns = start_ns + ((uint64_t)rt->config.timeout_ms * 1000000ull);
	for (;;)
	{
		now_ns = SparkGlm52Pp13LoopbackMonoNs();
		if (now_ns > deadline_ns)
			return -1;
		progress = 0u;
		progress |= SparkGlm52Pp13LoopbackAcceptInput(rt);
		progress |= SparkGlm52Pp13LoopbackConnectOutput(rt);
		progress |= SparkGlm52Pp13LoopbackFinishConnect(rt);
		progress |= SparkGlm52Pp13LoopbackStartOrigin(rt);
		read_status = SparkGlm52Pp13LoopbackReadInput(rt);
		if (read_status < 0)
			return -2;
		if (read_status > 0)
		{
			progress = 1u;
			queue_status = SparkGlm52Pp13LoopbackQueueOutput(rt);
			rt->input_offset = 0u;
			if (queue_status > 0)
			{
				rt->input_packet.completed_wall_ns = SparkGlm52Pp13LoopbackWallNs();
				rt->input_packet.completed_mono_ns = SparkGlm52Pp13LoopbackMonoNs();
				SparkGlm52Pp13LoopbackPrintComplete(&rt->input_packet);
				rt->completed_count += 1u;
				rt->launched = 0u;
				if (rt->completed_count >= rt->config.repeat_count)
					return 0;
				continue;
			}
			if (queue_status < 0)
				continue;
			progress = 1u;
		}
		write_status = SparkGlm52Pp13LoopbackWriteOutput(rt);
		if (write_status > 0)
			progress = 1u;
		if (progress == 0u)
		{
			now_ns = SparkGlm52Pp13LoopbackMonoNs();
			if (now_ns > deadline_ns)
				return -1;
			wait_ns = deadline_ns - now_ns;
			if (rt->next_connect_mono_ns != 0u &&
				rt->next_connect_mono_ns > now_ns &&
				rt->next_connect_mono_ns < deadline_ns)
				wait_ns = rt->next_connect_mono_ns - now_ns;
			if (wait_ns > 1000000000ull)
				wait_ns = 1000000000ull;
			SparkGlm52Pp13LoopbackWaitForEvents(rt,wait_ns);
		}
	}
}

int main(int argc,char **argv)
{
	SparkGlm52Pp13LoopbackRuntime rt;
	int32_t result;
	memset(&rt,0,sizeof(rt));
	rt.listen_fd = -1;
	rt.input_fd = -1;
	rt.output_fd = -1;
	if (SparkGlm52Pp13LoopbackParseArgs(&rt.config,argc,argv) < 0)
	{
		fprintf(stderr,"usage: %s --rank n [--origin-rank n] [--launch-origin] [--port-base n] [--timeout-ms n] [--sequence n] [--repeat-count n]\n",argv[0]);
		return 2;
	}
	rt.listen_fd = SparkGlm52Pp13LoopbackListen(rt.config.port_base + rt.config.rank_index);
	if (rt.listen_fd < 0)
	{
		fprintf(stderr,"loopback_listen_failed rank=%u port=%u\n",
			rt.config.rank_index,
			rt.config.port_base + rt.config.rank_index);
		return 3;
	}
	printf("loopback_ready rank=%u port=%u origin=%u launch=%u\n",
		rt.config.rank_index,
		rt.config.port_base + rt.config.rank_index,
		rt.config.origin_rank,
		rt.config.launch_origin);
	fflush(stdout);
	result = SparkGlm52Pp13LoopbackRun(&rt);
	if (rt.input_fd >= 0)
		close(rt.input_fd);
	if (rt.output_fd >= 0)
		close(rt.output_fd);
	if (rt.listen_fd >= 0)
		close(rt.listen_fd);
	if (result != 0)
		fprintf(stderr,"loopback_failed rank=%u result=%d\n",rt.config.rank_index,result);
	return result == 0 ? 0 : 1;
}

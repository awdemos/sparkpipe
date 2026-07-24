#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_collective.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Multi-rank loopback all-reduce: every rank contributes an exactly
// representable integer pattern, every rank must end with the bitwise
// identical elementwise sum, across degrees two, four, eight, and sixteen,
// at both a hidden-vector size and a multi-row size that exceeds socket
// buffering, and repeated runs must be deterministic.

#define SPARK_TEST_TPC_LARGE_ELEMENTS (16u * 6144u)

typedef struct SparkTestTpcThread
{
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint16_t port_base;
	uint64_t element_count;
	float *values;
	float *scratch;
	SparkStatus status;
	pthread_barrier_t *barrier;
} SparkTestTpcThread;

static void SparkTestTpcFill(float *values,uint64_t element_count,uint32_t tp_rank)
{
	uint64_t element_index;
	for (element_index = 0u; element_index < element_count; ++element_index)
		values[element_index] = (float)((tp_rank + 1u) * (element_index % 7u + 1u));
}

static void *SparkTestTpcMain(void *argument)
{
	SparkTestTpcThread *thread = (SparkTestTpcThread *)argument;
	SparkTpCollectiveConfig config;
	SparkTpCollective collective;
	uint32_t step_index;
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
	config.tp_degree = thread->tp_degree;
	config.tp_rank = thread->tp_rank;
	config.listen_port = (uint16_t)(thread->port_base + thread->tp_rank);
	config.connect_timeout_milli = 5000u;
	for (step_index = 0u; (thread->tp_degree >> (step_index + 1u)) != 0u; ++step_index)
	{
		uint32_t partner = thread->tp_rank ^ (1u << step_index);
		snprintf(config.peers[step_index].host_name,SPARK_TP_COLLECTIVE_HOST_NAME_BYTES,"127.0.0.1");
		config.peers[step_index].port = (uint16_t)(thread->port_base + partner);
	}
	thread->status = SparkTpCollectiveCreate(&config,&collective);
	pthread_barrier_wait(thread->barrier);
	if (thread->status != SPARK_STATUS_OK)
		return 0;
	SparkTestTpcFill(thread->values,thread->element_count,thread->tp_rank);
	thread->status = SparkTpCollectiveAllReduceSumF32(&collective,thread->values,thread->element_count,thread->scratch);
	pthread_barrier_wait(thread->barrier);
	SparkTpCollectiveDestroy(&collective);
	return 0;
}

static void SparkTestTpcRun(uint32_t tp_degree,uint16_t port_base,uint64_t element_count)
{
	pthread_t threads[16];
	SparkTestTpcThread contexts[16];
	pthread_barrier_t barrier;
	float *storage,*scratch_storage;
	uint32_t rank_index;
	uint64_t element_index;
	storage = (float *)malloc(tp_degree * element_count * sizeof(float));
	scratch_storage = (float *)malloc(tp_degree * element_count * sizeof(float));
	assert(storage != 0 && scratch_storage != 0);
	pthread_barrier_init(&barrier,0,tp_degree);
	for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
	{
		contexts[rank_index].tp_degree = tp_degree;
		contexts[rank_index].tp_rank = rank_index;
		contexts[rank_index].port_base = port_base;
		contexts[rank_index].element_count = element_count;
		contexts[rank_index].values = storage + rank_index * element_count;
		contexts[rank_index].scratch = scratch_storage + rank_index * element_count;
		contexts[rank_index].status = SPARK_STATUS_INVALID_ARGUMENT;
		contexts[rank_index].barrier = &barrier;
		assert(pthread_create(&threads[rank_index],0,SparkTestTpcMain,&contexts[rank_index]) == 0);
	}
	for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
		assert(pthread_join(threads[rank_index],0) == 0);
	pthread_barrier_destroy(&barrier);
	for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
		assert(contexts[rank_index].status == SPARK_STATUS_OK);
	// Every rank holds the exact expected sum, and every rank's buffer is
	// bitwise identical to rank zero's.
	for (element_index = 0u; element_index < element_count; ++element_index)
	{
		float expected = (float)((element_index % 7u + 1u) *
			(tp_degree * (tp_degree + 1u) / 2u));
		assert(storage[element_index] == expected);
	}
	for (rank_index = 1u; rank_index < tp_degree; ++rank_index)
		assert(memcmp(storage,storage + rank_index * element_count,element_count * sizeof(float)) == 0);
	free(storage);
	free(scratch_storage);
}

static void SparkTestTpcDegreeOneNoOp(void)
{
	SparkTpCollectiveConfig config;
	SparkTpCollective collective;
	float value = 42.0f;
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
	config.tp_degree = 1u;
	assert(SparkTpCollectiveCreate(&config,&collective) == SPARK_STATUS_OK);
	assert(SparkTpCollectiveAllReduceSumF32(&collective,&value,1u,0) == SPARK_STATUS_OK);
	assert(value == 42.0f);
	SparkTpCollectiveDestroy(&collective);
}

static void SparkTestTpcRejectsInvalidDegree(void)
{
	SparkTpCollectiveConfig config;
	SparkTpCollective collective;
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
	config.tp_degree = 3u;
	assert(SparkTpCollectiveCreate(&config,&collective) == SPARK_STATUS_INVALID_ARGUMENT);
	config.tp_degree = 4u;
	config.tp_rank = 4u;
	assert(SparkTpCollectiveCreate(&config,&collective) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
	SparkTestTpcDegreeOneNoOp();
	SparkTestTpcRejectsInvalidDegree();
	SparkTestTpcRun(2u,42100u,6144u);
	SparkTestTpcRun(4u,42120u,6144u);
	SparkTestTpcRun(8u,42140u,6144u);
	SparkTestTpcRun(16u,42160u,SPARK_TEST_TPC_LARGE_ELEMENTS);
	// Determinism across repeated runs on fresh ports.
	SparkTestTpcRun(8u,42200u,6144u);
	SparkTestTpcRun(8u,42220u,6144u);
	return 0;
}

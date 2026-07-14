#pragma once

#include <cuda_runtime_api.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sparkpipe/spark_status.h"

enum
{
	SparkGlm52ResidentPackIoCopyChunkBytes = 64u * 1024u * 1024u
};

static SparkStatus SparkGlm52ResidentPackIoCudaStatus(cudaError_t cuda_status)
{
	if ( cuda_status == cudaSuccess )
		return(SPARK_STATUS_OK);
	if ( cuda_status == cudaErrorMemoryAllocation )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_INTERNAL_ERROR);
}

static SparkStatus SparkGlm52ResidentPackIoCheckedAdd(uint64_t left,uint64_t right,uint64_t *sum_out)
{
	if ( sum_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( left > (UINT64_MAX - right) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*sum_out = (left + right);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ResidentPackIoRead(FILE *file,void *destination,uint64_t byte_count)
{
	if ( file == 0 || destination == 0 || byte_count > (uint64_t)((size_t)-1) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( byte_count == 0u )
		return(SPARK_STATUS_OK);
	if ( fread(destination,1u,(size_t)byte_count,file) != (size_t)byte_count )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ResidentPackIoSeek(FILE *file,uint64_t file_offset)
{
	if ( file == 0 || file_offset > (uint64_t)LONG_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( fseek(file,(long)file_offset,SEEK_SET) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ResidentPackIoFileSize(FILE *file,uint64_t *file_size_out)
{
	long original_offset,end_offset;
	if ( file == 0 || file_size_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	original_offset = ftell(file);
	if ( original_offset < 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( fseek(file,0,SEEK_END) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	end_offset = ftell(file);
	if ( end_offset < 0 || fseek(file,original_offset,SEEK_SET) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	*file_size_out = (uint64_t)end_offset;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ResidentPackIoLoadDeviceRegion(FILE *file,uint64_t offset,uint64_t bytes,void **device_pointer_out)
{
	uint8_t *host_buffer,*device_bytes;
	uint64_t copied_bytes,remaining_bytes;
	size_t chunk_bytes;
	SparkStatus status;
	if ( file == 0 || device_pointer_out == 0 || bytes == 0u || bytes > (uint64_t)((size_t)-1) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*device_pointer_out = 0;
	status = SparkGlm52ResidentPackIoCudaStatus(cudaMalloc(device_pointer_out,(size_t)bytes));
	if ( status != SPARK_STATUS_OK )
		return(status);
	host_buffer = (uint8_t *)malloc(SparkGlm52ResidentPackIoCopyChunkBytes);
	if ( host_buffer == 0 )
	{
		cudaFree(*device_pointer_out);
		*device_pointer_out = 0;
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkGlm52ResidentPackIoSeek(file,offset);
	device_bytes = (uint8_t *)(*device_pointer_out);
	copied_bytes = 0u;
	while ( status == SPARK_STATUS_OK && copied_bytes < bytes )
	{
		remaining_bytes = (bytes - copied_bytes);
		chunk_bytes = remaining_bytes > SparkGlm52ResidentPackIoCopyChunkBytes ? (size_t)SparkGlm52ResidentPackIoCopyChunkBytes : (size_t)remaining_bytes;
		status = SparkGlm52ResidentPackIoRead(file,host_buffer,(uint64_t)chunk_bytes);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52ResidentPackIoCudaStatus(cudaMemcpy(device_bytes + copied_bytes,host_buffer,chunk_bytes,cudaMemcpyHostToDevice));
		copied_bytes += (uint64_t)chunk_bytes;
	}
	free(host_buffer);
	if ( status != SPARK_STATUS_OK )
	{
		cudaFree(*device_pointer_out);
		*device_pointer_out = 0;
	}
	return(status);
}

static void SparkGlm52ResidentPackIoFreeDevice(void **device_pointer_cell)
{
	if ( device_pointer_cell != 0 && *device_pointer_cell != 0 )
	{
		cudaFree(*device_pointer_cell);
		*device_pointer_cell = 0;
	}
}

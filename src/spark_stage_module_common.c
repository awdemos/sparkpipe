/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_stage_module_common.h"

#include <stdlib.h>
#include <string.h>

SparkStatus SparkStageModuleCudaStatus(const char *module_tag, cudaError_t error, const char *site)
{
	if ( error == cudaSuccess )
		return(SPARK_STATUS_OK);
	fprintf(stderr,"%s cuda_error site=%s error=%s\n",module_tag,site,cudaGetErrorString(error));
	if ( error == cudaErrorMemoryAllocation )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_INTERNAL_ERROR);
}

SparkStatus SparkStageModuleEnvironmentText(const char *module_tag, const char *name, const char **value)
{
	const char *text = getenv(name);
	if ( text == 0 || text[0] == '\0' )
	{
		fprintf(stderr,"%s config_missing name=%s\n",module_tag,name);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	*value = text;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageModuleEnvironmentUnsigned(const char *module_tag, const char *name, uint32_t minimum, uint32_t maximum, uint32_t *value)
{
	const char *text;
	char *end;
	unsigned long parsed;
	SparkStatus status = SparkStageModuleEnvironmentText(module_tag,name,&text);
	if ( status != SPARK_STATUS_OK )
		return(status);
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( end == text || (end != 0 && *end != '\0') || parsed < (unsigned long)minimum || parsed > (unsigned long)maximum )
	{
		fprintf(stderr,"%s config_invalid name=%s value=%s allowed=[%u,%u]\n",module_tag,name,text,minimum,maximum);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	*value = (uint32_t)parsed;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageModuleEnvironmentUnsigned64(const char *module_tag, const char *name, uint64_t minimum, uint64_t maximum, uint64_t *value)
{
	const char *text;
	char *end;
	unsigned long long parsed;
	SparkStatus status = SparkStageModuleEnvironmentText(module_tag,name,&text);
	if ( status != SPARK_STATUS_OK )
		return(status);
	end = 0;
	parsed = strtoull(text,&end,10);
	if ( end == text || (end != 0 && *end != '\0') || parsed < minimum || parsed > maximum )
	{
		fprintf(stderr,"%s config_invalid name=%s value=%s allowed=[%llu,%llu]\n",module_tag,name,text,(unsigned long long)minimum,(unsigned long long)maximum);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	*value = (uint64_t)parsed;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageModuleRecordAllocation(SparkStageModuleLedger *ledger, void *pointer)
{
	if ( ledger->device_allocation_count >= SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS )
	{
		fprintf(stderr,"%s allocation_ledger_full count=%u\n",ledger->module_tag,ledger->device_allocation_count);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	ledger->device_allocations[ledger->device_allocation_count] = pointer;
	ledger->device_allocation_count++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageModuleDeviceAllocate(SparkStageModuleLedger *ledger, uint64_t bytes, void **pointer)
{
	SparkStatus status;
	void *allocation = 0;
	if ( bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageModuleCudaStatus(ledger->module_tag,cudaMalloc(&allocation,(size_t)bytes),"cudaMalloc");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleRecordAllocation(ledger,allocation);
	if ( status != SPARK_STATUS_OK )
	{
		cudaFree(allocation);
		return(status);
	}
	ledger->device_bytes_resident += bytes;
	*pointer = allocation;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageModuleDeviceAllocateZeroed(SparkStageModuleLedger *ledger, uint64_t bytes, void **pointer)
{
	SparkStatus status = SparkStageModuleDeviceAllocate(ledger,bytes,pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkStageModuleCudaStatus(ledger->module_tag,cudaMemset(*pointer,0,(size_t)bytes),"cudaMemset"));
}

void SparkStageModuleLedgerRelease(SparkStageModuleLedger *ledger)
{
	uint32_t index;
	for (index = 0; index < ledger->device_allocation_count; index++)
		cudaFree(ledger->device_allocations[index]);
	ledger->device_allocation_count = 0u;
	ledger->device_bytes_resident = 0u;
}

SparkStatus SparkStageModulePackRead(const char *module_tag, FILE *file, uint64_t offset, void *destination, uint64_t bytes)
{
	if ( fseeko(file,(off_t)offset,SEEK_SET) != 0 )
	{
		fprintf(stderr,"%s pack_seek_failed offset=%llu\n",module_tag,(unsigned long long)offset);
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( fread(destination,1,(size_t)bytes,file) != (size_t)bytes )
	{
		fprintf(stderr,"%s pack_read_failed offset=%llu bytes=%llu\n",module_tag,(unsigned long long)offset,(unsigned long long)bytes);
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageModuleLoadDeviceRegion(SparkStageModuleLedger *ledger, FILE *file, uint64_t offset, uint64_t bytes, void **pointer)
{
	SparkStatus status;
	void *device = 0;
	void *staging;
	uint64_t moved,chunk;
	status = SparkStageModuleDeviceAllocate(ledger,bytes,&device);
	if ( status != SPARK_STATUS_OK )
		return(status);
	staging = malloc(bytes < SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES ? (size_t)bytes : (size_t)SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES);
	if ( staging == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (moved = 0; moved < bytes; moved += chunk)
	{
		chunk = bytes - moved;
		if ( chunk > SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES )
			chunk = SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES;
		status = SparkStageModulePackRead(ledger->module_tag,file,offset + moved,staging,chunk);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(ledger->module_tag,cudaMemcpy((uint8_t *)device + moved,staging,(size_t)chunk,cudaMemcpyHostToDevice),"cudaMemcpy_h2d");
		if ( status != SPARK_STATUS_OK )
		{
			free(staging);
			return(status);
		}
	}
	free(staging);
	*pointer = device;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageModuleSlotClaim(atomic_uint *slot_states, uint32_t slot_count, uint32_t *slot_index)
{
	uint32_t index,expected;
	for (index = 0; index < slot_count; index++)
	{
		expected = SPARK_STAGE_MODULE_SLOT_FREE;
		if ( atomic_compare_exchange_strong(&slot_states[index],&expected,SPARK_STAGE_MODULE_SLOT_CLAIMED) )
		{
			*slot_index = index;
			return(SPARK_STATUS_OK);
		}
	}
	return(SPARK_STATUS_BUSY);
}

void SparkStageModuleSlotRelease(atomic_uint *slot_states, uint32_t slot_index)
{
	atomic_store(&slot_states[slot_index],SPARK_STAGE_MODULE_SLOT_FREE);
}

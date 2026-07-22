#ifndef SPARKPIPE_SPARK_STAGE_MODULE_COMMON_H
#define SPARKPIPE_SPARK_STAGE_MODULE_COMMON_H

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_status.h"

/*
 * Model-agnostic host substrate for resident decode stage modules, extracted
 * verbatim from the audited K3 module (2026-07-18) with the module name as a
 * tag so error lines stay attributable. Everything here is initialization
 * and control path; nothing touches the per-token hot path, which is where
 * the DRY-versus-performance line sits for this family.
 *
 * The rules these functions encode:
 * - configuration comes from the environment and every value is REQUIRED; a
 *   missing or unparsable variable is a hard failure, because a silently
 *   defaulted configuration produces a model that runs and is wrong
 * - device allocation happens only through the ledger, only during
 *   initialization, and every pointer is recorded so Destroy releases the
 *   node without a leak
 * - pack tensors stage through a bounded host buffer so a multi-gigabyte
 *   tensor never needs a host copy of its own size
 * - pipeline slots are claimed with a compare-and-swap so concurrent frames
 *   can never share scratch
 *
 * C only (stdatomic): included by module .c translation units, never by the
 * CUDA TU.
 */

#define SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS 4096u
#define SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES (64ull * 1024u * 1024u)
#define SPARK_STAGE_MODULE_SLOT_FREE 0u
#define SPARK_STAGE_MODULE_SLOT_CLAIMED 1u

typedef struct SparkStageModuleLedger
{
	const char *module_tag;
	void *device_allocations[SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS];
	uint32_t device_allocation_count;
	uint64_t device_bytes_resident;
} SparkStageModuleLedger;

SparkStatus SparkStageModuleCudaStatus(const char *module_tag, cudaError_t error, const char *site);
SparkStatus SparkStageModuleEnvironmentText(const char *module_tag, const char *name, const char **value);
SparkStatus SparkStageModuleEnvironmentUnsigned(const char *module_tag, const char *name, uint32_t minimum, uint32_t maximum, uint32_t *value);
SparkStatus SparkStageModuleEnvironmentUnsigned64(const char *module_tag, const char *name, uint64_t minimum, uint64_t maximum, uint64_t *value);
SparkStatus SparkStageModuleDeviceAllocate(SparkStageModuleLedger *ledger, uint64_t bytes, void **pointer);
SparkStatus SparkStageModuleDeviceAllocateZeroed(SparkStageModuleLedger *ledger, uint64_t bytes, void **pointer);
void SparkStageModuleLedgerRelease(SparkStageModuleLedger *ledger);
SparkStatus SparkStageModulePackRead(const char *module_tag, FILE *file, uint64_t offset, void *destination, uint64_t bytes);
SparkStatus SparkStageModuleLoadDeviceRegion(SparkStageModuleLedger *ledger, FILE *file, uint64_t offset, uint64_t bytes, void **pointer);
SparkStatus SparkStageModuleSlotClaim(atomic_uint *slot_states, uint32_t slot_count, uint32_t *slot_index);
void SparkStageModuleSlotRelease(atomic_uint *slot_states, uint32_t slot_index);

#endif

#ifndef SPARKPIPE_SPARK_STATUS_H
#define SPARKPIPE_SPARK_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkStatus
{
    SPARK_STATUS_OK = 0,
    SPARK_STATUS_INVALID_ARGUMENT,
    SPARK_STATUS_CAPACITY_EXCEEDED,
    SPARK_STATUS_NOT_FOUND,
    SPARK_STATUS_IO_ERROR,
    SPARK_STATUS_PARSE_ERROR,
    SPARK_STATUS_SCHEMA_ERROR,
    SPARK_STATUS_HASH_MISMATCH,
    SPARK_STATUS_MODULE_NOT_VALIDATED,
    SPARK_STATUS_VALIDATION_FAILED,
    SPARK_STATUS_ABI_MISMATCH,
    SPARK_STATUS_TARGET_MISMATCH,
    SPARK_STATUS_COMPILER_ERROR,
    SPARK_STATUS_DRIVER_LOAD_ERROR,
    SPARK_STATUS_ROUTE_NOT_FOUND,
    SPARK_STATUS_BUSY,
    SPARK_STATUS_DUPLICATE,
    SPARK_STATUS_INTERNAL_ERROR,
    SPARK_STATUS_PENDING,
    SPARK_STATUS_UNSUPPORTED
} SparkStatus;

const char *SparkStatusToString(SparkStatus status);

#ifdef __cplusplus
}
#endif


#include <stdint.h>
#include <stdio.h>

// Shared micro-helpers, extracted from three and four near-identical
// copies respectively (the duplication instrument found them at 0.97+
// similarity). One definition; everyone includes this header already.
static inline SparkStatus SparkReportError(char *error_buffer, uint32_t error_buffer_bytes, SparkStatus status, const char *message)
{
	if ( error_buffer != 0 && error_buffer_bytes != 0u )
	{
		if ( message == 0 )
			error_buffer[0] = '\0';
		else
			(void)snprintf(error_buffer, error_buffer_bytes, "%s", message);
	}
	return status;
}

static inline uint32_t SparkCeilDivU32(uint32_t numerator, uint32_t denominator)
{
	return denominator == 0u ? 0u : (numerator + denominator - 1u) / denominator;
}

static inline uint64_t SparkCeilDivU64(uint64_t numerator, uint64_t denominator)
{
	return denominator == 0u ? 0u : (numerator + denominator - 1u) / denominator;
}

#endif

// The kernel's swizzle chunk count, exposed to host code that cannot include a
// .cuh. spark_lm_group_gemm.cuh static_asserts against this, so the two cannot
// drift apart silently.
#ifndef SPARK_LM_GROUP_GEMM_SWIZZLE_CONTRACT_H
#define SPARK_LM_GROUP_GEMM_SWIZZLE_CONTRACT_H
#define SPARK_LM_GROUP_GEMM_SWIZZLE_CHUNKS_CONTRACT 8u
#endif

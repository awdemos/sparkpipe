#pragma once
#include <stdint.h>
#include <string.h>
#include <cmath>
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __restrict__
#define __grid_constant__
#define __align__(x)
#define __launch_bounds__(...)
struct dim3s { unsigned x,y,z; };
static dim3s threadIdx,blockIdx,blockDim,gridDim;
static inline void __syncthreads(){}
template<class T> static inline T __ldg(const T*p){return *p;}
static inline unsigned __float_as_uint(float f){unsigned u;memcpy(&u,&f,4);return u;}
static inline float __uint_as_float(unsigned u){float f;memcpy(&f,&u,4);return f;}
static inline unsigned long long __cvta_generic_to_shared(void*p){return (unsigned long long)p;}
static inline float fmaf_(float a,float b,float c){return a*b+c;}
typedef unsigned char __nv_fp8_storage_t;

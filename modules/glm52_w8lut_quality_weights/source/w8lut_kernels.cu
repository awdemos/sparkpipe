// w8lut_kernels.cu — W8LUT v2 CUDA path for sparkpipe expert GEMMs
// COMPILE STATUS: written against CUDA 12 + cuda_bf16.h, NOT compiled by author
// (no GPU/nvcc in authoring environment). G1/G2 in test_w8lut_gpu.cu gate it.
// w8lut_dev_decode1 MUST stay bit-identical to w8lut_decode1 in w8lut.h; G1 enforces.
#include <stdint.h>
#include <cuda_bf16.h>

#define W8LUT_THREADS 256

static __device__ __forceinline__ uint16_t w8lut_dev_decode1(uint16_t e0,uint8_t c)
{
	if ( (c & 0x7F) == 0 )
		return((uint16_t)((uint16_t)(c & 0x80) << 8));
	return((uint16_t)(((uint16_t)(c & 0x80) << 8) | ((uint16_t)(e0 + ((c >> 4) & 7)) << 7) | ((uint16_t)(c & 0xF) << 3)));
}

static __device__ __forceinline__ float w8lut_dev_bf2f(uint16_t b)
{
	__nv_bfloat16_raw r;
	r.x = b;
	return(__bfloat162float(*(__nv_bfloat16 *)&r));
}

// bulk decode codes -> BF16 buffer (prefill path: expand then cublasLt BF16 GEMM)
extern "C" __global__ void w8lut_expand(const uint8_t *codes,uint16_t e0,uint16_t *out,uint32_t n)
{
	uint32_t i = ((blockIdx.x * blockDim.x) + threadIdx.x),stride = (gridDim.x * blockDim.x);
	for (; i<n; i+=stride)
		out[i] = w8lut_dev_decode1(e0,codes[i]);
}

// fused decode + GEMV for the decode phase: y[M,N] = x[M,K] * W[N,K]^T
// one block per output column n; weight row decoded to SMEM once, reused for all M
// deterministic: fixed 256-thread strided partials + fixed halving tree, no atomics
extern "C" __global__ void w8lut_gemv(const uint8_t *codes,uint16_t e0,const uint16_t *x,float *y,uint32_t M,uint32_t N,uint32_t K)
{
	extern __shared__ float smem[];
	float *wrow = smem,*red = (smem + K);
	uint32_t n = blockIdx.x,tid = threadIdx.x,k,m,s;
	float partial;
	if ( n >= N )
		return;
	for (k=tid; k<K; k+=W8LUT_THREADS)
		wrow[k] = w8lut_dev_bf2f(w8lut_dev_decode1(e0,codes[(n * K) + k]));
	__syncthreads();
	for (m=0; m<M; m++)
	{
		partial = 0.0f;
		for (k=tid; k<K; k+=W8LUT_THREADS)
			partial += (wrow[k] * w8lut_dev_bf2f(x[(m * K) + k]));
		red[tid] = partial;
		__syncthreads();
		for (s=(W8LUT_THREADS >> 1); s>0; s>>=1)
		{
			if ( tid < s )
				red[tid] += red[tid + s];
			__syncthreads();
		}
		if ( tid == 0 )
			y[(m * N) + n] = red[0];
		__syncthreads();
	}
}

// host-side launch helpers (thin, so sparkpipe integration is one call per tensor)
extern "C" int32_t w8lut_expand_launch(const uint8_t *dcodes,uint16_t e0,uint16_t *dout,uint32_t n,void *stream)
{
	uint32_t blocks = ((n + W8LUT_THREADS - 1) / W8LUT_THREADS);
	if ( blocks > 65535 )
		blocks = 65535;
	w8lut_expand<<<blocks,W8LUT_THREADS,0,(cudaStream_t)stream>>>(dcodes,e0,dout,n);
	return((int32_t)cudaGetLastError() == 0 ? 0 : -51);
}

extern "C" int32_t w8lut_gemv_launch(const uint8_t *dcodes,uint16_t e0,const uint16_t *dx,float *dy,uint32_t M,uint32_t N,uint32_t K,void *stream)
{
	uint32_t shmem = (((K + W8LUT_THREADS) * 4));
	if ( K > 8192 || M == 0 || N == 0 )
		return(-52);
	w8lut_gemv<<<N,W8LUT_THREADS,shmem,(cudaStream_t)stream>>>(dcodes,e0,dx,dy,M,N,K);
	return((int32_t)cudaGetLastError() == 0 ? 0 : -53);
}

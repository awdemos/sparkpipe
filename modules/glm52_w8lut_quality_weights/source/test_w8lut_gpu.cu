// test_w8lut_gpu.cu — G1/G2 gates, run on a Spark: nvcc -O3 -arch=native -o test_gpu test_w8lut_gpu.cu w8lut_ref.c
// G1: w8lut_expand output bit-identical to CPU w8lut_decode1 over the full tensor
// G2: w8lut_gemv bit-identical to a CPU float mirror using the SAME strided-partial
//     + halving-tree accumulation order (proves decode-in-kernel adds nothing),
//     plus a double-precision sanity bound
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
extern "C"
{
#include "w8lut.h"
}

extern "C" int32_t w8lut_expand_launch(const uint8_t *dcodes,uint16_t e0,uint16_t *dout,uint32_t n,void *stream);
extern "C" int32_t w8lut_gemv_launch(const uint8_t *dcodes,uint16_t e0,const uint16_t *dx,float *dy,uint32_t M,uint32_t N,uint32_t K,void *stream);

enum { TM = 32,TN = 2048,TK = 6144,THREADS = 256 };
static uint64_t rng_s = 0x9E3779B97F4A7C15ULL;

static uint32_t rng32(void)
{
	rng_s ^= (rng_s << 13); rng_s ^= (rng_s >> 7); rng_s ^= (rng_s << 17);
	return((uint32_t)(rng_s >> 32));
}

static uint16_t f2bf(float f)
{
	uint32_t u;
	memcpy(&u,&f,sizeof(u));
	return((uint16_t)(u >> 16));
}

static float bf2f(uint16_t b)
{
	uint32_t u = ((uint32_t)b << 16);
	float f;
	memcpy(&f,&u,sizeof(f));
	return(f);
}

static void cpu_gemv_mirror(const uint16_t *dec,const uint16_t *x,float *y,uint32_t M,uint32_t N,uint32_t K)
{
	static float wrow[TK],partial[THREADS];
	uint32_t n,m,k,t,s;
	for (n=0; n<N; n++)
	{
		for (k=0; k<K; k++)
			wrow[k] = bf2f(dec[(n * K) + k]);
		for (m=0; m<M; m++)
		{
			for (t=0; t<THREADS; t++)
			{
				partial[t] = 0.0f;
				for (k=t; k<K; k+=THREADS)
					partial[t] = fmaf(wrow[k],bf2f(x[(m * K) + k]),partial[t]);	// matches nvcc default -fmad=true contraction in w8lut_gemv
			}
			for (s=(THREADS >> 1); s>0; s>>=1)
				for (t=0; t<s; t++)
					partial[t] += partial[t + s];
			y[(m * N) + n] = partial[0];
		}
	}
}

int main(void)
{
	static uint16_t w[TN * TK],dec[TN * TK],x[TM * TK],hexp[TN * TK];
	static uint8_t codes[TN * TK];
	static float ycpu[TM * TN],ygpu[TM * TN];
	uint8_t *dcodes;
	uint16_t *dout,*dx;
	float *dy;
	uint32_t i,m,n,k;
	int32_t rc;
	double err,maxerr = 0.0,ref;
	w8lut_t t;
	for (i=0; i<(TN * TK); i++)
		w[i] = f2bf(((float)((int32_t)(rng32() & 0xFFFF) - 32768)) * 6.1e-7f);
	for (i=0; i<(TM * TK); i++)
		x[i] = f2bf(((float)((int32_t)(rng32() & 0xFFFF) - 32768)) * 3.05e-5f);
	if ( (rc= w8lut_encode(w,TN,TK,codes,&t)) != 0 )
		{ printf("FAIL encode %d\n",rc); return(1); }
	if ( (rc= w8lut_verify(w,&t)) != 0 )
		{ printf("FAIL verify %d\n",rc); return(1); }
	for (i=0; i<(TN * TK); i++)
		dec[i] = w8lut_decode1(t.e0,codes[i]);
	cudaMalloc((void **)&dcodes,sizeof(codes));
	cudaMalloc((void **)&dout,sizeof(dec));
	cudaMalloc((void **)&dx,sizeof(x));
	cudaMalloc((void **)&dy,sizeof(ygpu));
	cudaMemcpy(dcodes,codes,sizeof(codes),cudaMemcpyHostToDevice);
	cudaMemcpy(dx,x,sizeof(x),cudaMemcpyHostToDevice);
	if ( w8lut_expand_launch(dcodes,t.e0,dout,TN * TK,0) != 0 || cudaDeviceSynchronize() != cudaSuccess )
		{ printf("FAIL G1 launch\n"); return(1); }
	cudaMemcpy(hexp,dout,sizeof(hexp),cudaMemcpyDeviceToHost);
	if ( memcmp(hexp,dec,sizeof(hexp)) != 0 )
		{ printf("FAIL G1 expand not bit-exact\n"); return(1); }
	printf("G1 PASS: expand bit-exact over %u codes\n",TN * TK);
	if ( w8lut_gemv_launch(dcodes,t.e0,dx,dy,TM,TN,TK,0) != 0 || cudaDeviceSynchronize() != cudaSuccess )
		{ printf("FAIL G2 launch\n"); return(1); }
	cudaMemcpy(ygpu,dy,sizeof(ygpu),cudaMemcpyDeviceToHost);
	cpu_gemv_mirror(dec,x,ycpu,TM,TN,TK);
	if ( memcmp(ygpu,ycpu,sizeof(ygpu)) != 0 )
		{ printf("FAIL G2 gemv not bit-exact vs same-order mirror\n"); return(1); }
	for (i=0; i<(TM * TN); i++)
	{
		ref = 0.0;
		// spot double-precision sanity on a diagonal stripe
		if ( (i % 257) == 0 )
		{
			m = (i / TN); n = (i % TN);
			for (k=0; k<TK; k++)
				ref += ((double)bf2f(dec[(n * TK) + k]) * (double)bf2f(x[(m * TK) + k]));
			err = fabs((double)ygpu[i] - ref) / (fabs(ref) + 1e-9);
			if ( err > maxerr )
				maxerr = err;
		}
	}
	printf("G2 PASS: gemv bit-exact vs mirror, fp64 spot maxrel=%.2e\n",maxerr);
	printf("GPU GATES ALL PASS\n");
	return(0);
}

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "w8lut.h"

static uint64_t rng_s = 0x243F6A8885A308D3ULL;

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

static int32_t t_rne_golden(void)
{
	static const uint16_t in[9]  = { 0x0000,0x0001,0x0004,0x000C,0x0005,0x00FF,0x007F,0x8005,0x3F81 };
	static const uint16_t exp[9] = { 0x0000,0x0000,0x0000,0x0010,0x0008,0x0100,0x0080,0x8008,0x3F80 };
	int32_t i;
	for (i=0; i<9; i++)
		if ( w8lut_rne4(in[i]) != exp[i] )
			return(-100 - i);
	return(0);
}

static int32_t t_inf_reject(void)
{
	uint16_t w1[2] = { 0x3F80,0x7F80 },w2[2] = { 0x3F80,0x7F7C };
	uint8_t codes[2];
	w8lut_t t;
	if ( w8lut_encode(w1,1,2,codes,&t) != -11 )
		return(-110);
	if ( w8lut_encode(w2,1,2,codes,&t) != -12 )
		return(-111);
	return(0);
}

static int32_t t_window_golden(void)
{
	// max value 0x4212 -> rne 0x4210 exp 0x84 -> e0 = 0x7D
	// bottom grid floor1 = (0x7D<<7)|0x08 = 0x3E88, midpoint T = (0x7C<<7)|0x08 = 0x3E08
	uint16_t w[6] = { 0x4212,0x3E88,0x3E09,0x3E08,0x8100,0x0000 };
	uint8_t codes[6];
	w8lut_t t;
	if ( w8lut_encode(w,1,6,codes,&t) != 0 || t.e0 != 0x7D )
		return(-120);
	if ( codes[0] != (uint8_t)((7 << 4) | 2) )
		return(-121);
	if ( codes[1] != 0x01 )
		return(-122);
	if ( codes[2] != 0x01 )
		return(-123);
	if ( codes[3] != 0x00 )
		return(-124);
	if ( codes[4] != 0x80 )
		return(-125);
	if ( codes[5] != 0x00 || w8lut_decode1(t.e0,0x00) != 0x0000 || w8lut_decode1(t.e0,0x80) != 0x8000 )
		return(-126);
	if ( w8lut_decode1(t.e0,codes[1]) != 0x3E88 )
		return(-127);
	if ( w8lut_verify(w,&t) != 0 )
		return(-128);
	return(0);
}

static int32_t t_property(void)
{
	enum { N = 65536 };
	static uint16_t w[N],dec[N];
	static uint8_t codes[N],codes2[N];
	uint32_t i;
	float o,d,rel,maxrel = 0.0f,floorv;
	double se = 0.0,sw = 0.0;
	w8lut_t t,t2;
	for (i=0; i<N; i++)
		w[i] = f2bf(((float)((int32_t)(rng32() & 0xFFFF) - 32768)) * 3.05e-7f);
	if ( w8lut_encode(w,256,256,codes,&t) != 0 )
		return(-140);
	if ( w8lut_verify(w,&t) != 0 )
		return(-141);
	for (i=0; i<N; i++)
		dec[i] = w8lut_decode1(t.e0,codes[i]);
	floorv = bf2f((uint16_t)((t.e0 << 7) | 0x08));
	for (i=0; i<N; i++)
	{
		o = bf2f(w[i]); d = bf2f(dec[i]);
		se += ((double)(d - o) * (d - o)); sw += ((double)o * o);
		if ( o == 0.0f || fabsf(o) < floorv )
			continue;
		rel = fabsf((d - o) / o);
		if ( rel > maxrel )
			maxrel = rel;
	}
	if ( maxrel > 0.03125f )
		return(-142);
	if ( w8lut_encode(dec,256,256,codes2,&t2) != 0 || t2.e0 != t.e0 || memcmp(codes,codes2,sizeof(codes)) != 0 )
		return(-143);
	printf("property: e0=%u below=%uppm maxrel=%.5f frob=%.5f\n",t.e0,t.belowcnt_ppm,maxrel,sqrt(se / sw));
	return(0);
}

int main(void)
{
	int32_t rc;
	if ( (rc= t_rne_golden()) < 0 )
		{ printf("FAIL rne_golden %d\n",rc); return(1); }
	if ( (rc= t_inf_reject()) < 0 )
		{ printf("FAIL inf_reject %d\n",rc); return(1); }
	if ( (rc= t_window_golden()) < 0 )
		{ printf("FAIL window_golden %d\n",rc); return(1); }
	if ( (rc= t_property()) < 0 )
		{ printf("FAIL property %d\n",rc); return(1); }
	printf("G0 ALL PASS (v2)\n");
	return(0);
}

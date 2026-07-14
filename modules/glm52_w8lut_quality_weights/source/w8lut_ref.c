#include "w8lut.h"

static int32_t w8lut_maxexp(const uint16_t *w,uint32_t n,uint16_t *e7p)
{
	uint32_t i;
	uint16_t e,e7 = 0;
	for (i=0; i<n; i++)
	{
		if ( ((w[i] >> 7) & 0xFF) == 0xFF )
			return(-11);
		e = (uint16_t)((w8lut_rne4(w[i]) >> 7) & 0xFF);
		if ( e == 0xFF )
			return(-12);
		if ( e > e7 )
			e7 = e;
	}
	*e7p = e7;
	return(0);
}

static uint8_t w8lut_enc1(uint16_t b,uint16_t e0)
{
	uint16_t r = w8lut_rne4(b),e = (uint16_t)((r >> 7) & 0xFF),m = (uint16_t)((r >> 3) & 0xF);
	uint16_t s = (uint16_t)((b >> 15) << 7),mag = (uint16_t)(b & 0x7FFF),T;
	if ( e > e0 || (e == e0 && m > 0) )
		return((uint8_t)(s | ((e - e0) << 4) | m));
	T = (uint16_t)(e0 > 0 ? (((e0 - 1) << 7) | 0x08) : 0x04);
	if ( mag > T )
		return((uint8_t)(s | 0x01));
	return((uint8_t)s);
}

int32_t w8lut_encode(const uint16_t *w,uint32_t rows,uint32_t cols,uint8_t *codes,w8lut_t *out)
{
	uint32_t i,below = 0,n = (rows * cols);
	uint16_t e7,e0;
	int32_t rc;
	if ( w == 0 || codes == 0 || out == 0 || n == 0 )
		return(-1);
	if ( (rc= w8lut_maxexp(w,n,&e7)) < 0 )
		return(rc);
	e0 = (uint16_t)(e7 >= 7 ? e7 - 7 : 0);
	for (i=0; i<n; i++)
	{
		codes[i] = w8lut_enc1(w[i],e0);
		if ( (uint16_t)((w8lut_rne4(w[i]) >> 7) & 0xFF) < e0 && (w[i] & 0x7FFF) != 0 )
			below++;
	}
	out->rows = rows; out->cols = cols; out->e0 = e0;
	out->belowcnt_ppm = (uint32_t)(((uint64_t)below * 1000000) / n);
	out->codes = codes;
	return(0);
}

int32_t w8lut_verify(const uint16_t *w,const w8lut_t *t)
{
	uint32_t i,n = (t->rows * t->cols);
	uint16_t d,r,e;
	for (i=0; i<n; i++)
	{
		if ( t->codes[i] != w8lut_enc1(w[i],t->e0) )
			return(-31);
		d = w8lut_decode1(t->e0,t->codes[i]);
		r = w8lut_rne4(w[i]);
		e = (uint16_t)((r >> 7) & 0xFF);
		if ( (e > t->e0 || (e == t->e0 && ((r >> 3) & 0xF) != 0)) && d != r )
			return(-32);
	}
	return(0);
}

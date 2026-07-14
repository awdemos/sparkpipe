#pragma once
#include <stdint.h>

// W8LUT v2: per-tensor exponent-biased 1-3-4 minifloat for BF16 weight tensors
// code byte: [7]=sign, [6:4]=eidx, [3:0]=4-bit RNE mantissa
// decode: (c & 0x7F) == 0 -> signed zero, else bf16 = s<<15 | (e0+eidx)<<7 | man<<3
// e0 chosen so e0+7 == max exponent of RNE-rounded tensor: nothing clips above, ever
// below-window values round to nearest of { 0, (eidx=0,man=1..15) grid }
// inf/nan in source is a conversion error, never encoded
#define W8LUT_MAGIC 0x57384C32

typedef struct
{
	uint32_t rows,cols,belowcnt_ppm;
	uint16_t e0,pad;
	const uint8_t *codes;
} w8lut_t;

static inline uint16_t w8lut_rne4(uint16_t b) { return((uint16_t)((b + 0x0003 + ((b >> 3) & 1)) & 0xFFF8)); }

static inline uint16_t w8lut_decode1(uint16_t e0,uint8_t c)
{
	if ( (c & 0x7F) == 0 )
		return((uint16_t)((uint16_t)(c & 0x80) << 8));
	return((uint16_t)(((uint16_t)(c & 0x80) << 8) | ((uint16_t)(e0 + ((c >> 4) & 7)) << 7) | ((uint16_t)(c & 0xF) << 3)));
}

int32_t w8lut_encode(const uint16_t *w,uint32_t rows,uint32_t cols,uint8_t *codes,w8lut_t *out);
int32_t w8lut_verify(const uint16_t *w,const w8lut_t *t);

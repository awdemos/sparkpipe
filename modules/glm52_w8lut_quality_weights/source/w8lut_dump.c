#include <stdio.h>
#include <stdlib.h>
#include "w8lut.h"

int main(int argc,char **argv)
{
	static uint16_t w[1 << 20];
	static uint8_t codes[1 << 20];
	uint32_t n;
	int32_t rc;
	w8lut_t t;
	FILE *fp;
	if ( argc != 3 )
		return(1);
	fp = fopen(argv[1],"rb");
	n = (uint32_t)(fread(w,sizeof(w[0]),(sizeof(w) / sizeof(w[0])),fp));
	fclose(fp);
	if ( (rc= w8lut_encode(w,1,n,codes,&t)) != 0 )
		{ printf("encode rc=%d\n",rc); return(2); }
	if ( (rc= w8lut_verify(w,&t)) != 0 )
		{ printf("verify rc=%d\n",rc); return(3); }
	fp = fopen(argv[2],"wb");
	fwrite(&t.e0,sizeof(t.e0),1,fp);
	fwrite(codes,sizeof(codes[0]),n,fp);
	fclose(fp);
	printf("ok n=%u e0=%u below=%uppm\n",n,t.e0,t.belowcnt_ppm);
	return(0);
}

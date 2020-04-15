#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "blis.h"

#define max(a,b) (((a)>(b ))?( a):(b))
#define min(a,b) (((a)<(b ))?( a):(b))

void convolutionNaive(const int h, const int w, const int c,const int b,const float* In,
					  const int kh,const int kw, const int kn, const float* F, 
					  float* Out, const int stride);
void im2Col(const int h, const int w, const int c, const int b,const float* In,
			const int kh, const int kw, const int stride,float* Out);
void padMatrix();

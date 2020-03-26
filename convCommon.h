#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "blis.h"

void convolutionNaive(const int h, const int w, const int c,const int b,const float* In,
					  const int kh,const int kw, const int kn, const float* F, 
					  float* Out, const int stride);
void im2Col(const int h, const int w, const int c, const int b,const float* In,
			const int kh, const int kw, const int stride,float* Out);
void padMatrix();

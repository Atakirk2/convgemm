/**Common convolution code
 * 
 * This file contains the declaration of convolution related functions used  by several sources in the project.
 * 
 * @author P. San Juan
 * @date 04/2020
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <blis.h>

#ifndef maxmin
    #define max(a, b) (((a)>(b))?(a):(b))
    #define min(a, b) (((a)<(b))?(a):(b))
#endif

void convolutionNaive(const int h, const int w, const int c,const int b,const float* In,
					  const int kh,const int kw, const int kn, const float* F, 
					  float* Out, const int stride);
void im2Col(const int h, const int w, const int c, const int b,const float* In,
			const int kh, const int kw, const int stride,float* Out);
void col2Im(const int h, const int w, const int c, const int b,const float* mat, const int kh, const int kw, const int stride, const int pad, float* Im);
void padMatrix(const int m, const int n, const int c, const int b, const float* In , const int pad, float * padM);

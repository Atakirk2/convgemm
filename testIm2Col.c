#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemmConv.h"
#include "blis.h"



#define EPS 1e16

void convolutionGEMMImplicit()
{
    
}

void im2Col(const int h, const int w, const int c, const int b,const float* In, const int kh, const int kw, const int stride,float* Out)
{
   /*     cdef int c, ii, jj, row, yy, xx, i, col

    #for c in range(C):
    for c in prange(C, nogil=True):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]*/
   int ic, ikh, ikw, ih, iw, ib,
       row,col;
   
    
    for(ib = 0;ib < b; ib++)    
       for( ic = 0; ic < c; ic++)
            for(iw = 0; iw < w; iw++)                
                for(ih = 0; ih < h; ih++)
                {
                    col = ib *w * h + iw * h + ih;
                    for(ikw = 0; ikw < kw; ikw++)
                        for(ikh = 0; ikh < kh; ikh++)
                        {
                            row = ic *kw*kh + ikw * kh + ikh; 
                            //printf("Writing into Out[%d,%d] from In[%d,%d,%d,%d]\n",
                              //    row,col,ib,ic, (iw * stride + ikw),(stride * ih + ikh));
                            Out[row + col * c*kw*kh] = In[ib * c*h*w + ic * h*w +
                                                        (iw * stride + ikw) * h + (stride * ih + ikh)];
                        }
                }
}

void convolutionNaive(const int h, const int w, const int c,const int b,const float* In,const int kh,const int kw, const int kn, const float* F, float* Out, const int stride,const int pad)
{
    
/*    def convolution2d_naive(input, weights, bias, p=0, s=1):
    h, w, ci, b    = input.shape
    co, kh, kw, ci = weights.shape
    ho = int((h + 2 * p - kh) / s + 1)
    wo = int((w + 2 * p - kw) / s + 1)
    input = input.transpose(3, 2, 0, 1) # b, c, h, w, this is needed for padding
    input_padded = np.pad(input, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant').transpose(2, 3, 1, 0)
    out = np.zeros((ho, wo, co, b))
    for b_ in range(b):
        for co_ in range(co):
            for ci_ in range(ci):
                for h_ in range(ho):
                    for w_ in range(wo):
                        for kh_ in range(kh):
                            for kw_ in range(kw):                        
                                out[h_, w_, co_, b_] += input_padded[h_ * s + kh_, w_ * s + kw_, ci_, b_] * weights[co_, kh_, kw_, ci_]
            out[..., co_, b_] += bias[co_]
    return out
    */
}

void padMatrix()
{
    
}

int compareMatrix(const int m,const int n, double* M, const int ldm, double* M2, const int ldm2){
    
    int equal = 1;
    double norm = 0;
    
    bli_dsubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,M,1,ldm,M2,1,ldm2);
    bli_dnormfv(m*n,M2,1,&norm);

    if (norm > EPS)
        equal = 0;
    
    return equal;
}

int main( int argc, char** argv )
{
    double tConv = 0.0, tIm2Col = 0.0, tIm2ColGemm = 0, tImp = 0.0, tIni,
            gflopsConv, gflopsIm2Col, gflopsImp;
            
    float ONE = 1, ZERO = 0;
    
    int i, m, n, k, 
        repe, //repetitions
        iZERO = 0;
        
     int h,w,c,b, //input dims
         ho,wo, //oputput dimms
         kh,kw, kn, //kernel dimms
         stride,pad; //algorithm parameters
        
    
    float *F, *In, 
         *OutConv, *OutI2c, *OutImp,
         *Aux;

    
    if (argc != 11)
    {
        printf("Comparison of Convolution implementations.\n");
        printf("\th,W,c: Imput tensor dimensions dimensions.\n");
        printf("\tb: Batch size.\n");
        printf("\tkh, kw: Kernel dimensions.\n");
        printf("\tkn: Kernel number.\n");
        printf("\tStride: kernel aplication stride.\n");
        printf("\tpad: Imput 0 padding.\n");
        printf("\trepe: number of repetitions of the test.\n");
        printf("Usage: %s <h> <w> <c> <b> <kh> <kw> <kn> <stride> <pad> <repe>\n", argv[0]);
        return -1;
    }

    h =atoi(argv[1]);  
    w =atoi(argv[2]);
    c = atoi(argv[3]);
    b  =atoi(argv[4]);
    kh  =atoi(argv[5]);
    kw  =atoi(argv[6]);
    kn  =atoi(argv[7]);
    stride =atoi(argv[8]);
    pad =atoi(argv[9]);
    repe =atoi(argv[10]);
                
    
    In = (float*) malloc(h*w*c *b * sizeof(float));
    F = (float*) malloc(kh*kw*c *kn * sizeof(float));

    ho = floor((h - kh + 2 * pad) / stride + 1);
    wo = floor((w - kw + 2 * pad) / stride + 1);
    
    Aux = (float*) malloc(c*kh*kw * ho*wo*b * sizeof(float));
    OutConv = (float*) malloc(ho*wo*kn*b * sizeof(float));
    OutI2c= (float*) malloc(ho*wo*kn*b * sizeof(float));
    OutImp = (float*) malloc(ho*wo*kn*b * sizeof(float));
    
    for(i = 0; i <  repe; i++)
    {
        //Timing naive convolution
        tIni = bli_clock();
        convolutionNaive(h,w,c,b,In,kh,kw,kn,F,OutConv, stride,pad);
        tConv += bli_clock() - tIni;
        
        //Timing im2col +gemm
        tIni = bli_clock();
        im2Col (ho,wo,c,b,In,kh,kw, stride,Aux);
        tIm2Col += bli_clock() -tIni;
        bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,kn,ho*wo*b,kh*kw*c,&ONE,F,1,kn,Aux,1,kh*kw*c,&ZERO,OutI2c,1,kn);
        tIm2ColGemm += bli_clock() -tIni;
        
        
        
        //Timing implicint gemm
        tIni = bli_clock();
        //gemm_conv
        tImp += bli_clock() -tIni;

    }
    
    tConv /=repe;
    tIm2Col/=repe;
    tIm2ColGemm/=repe;
    tImp/=repe;
    

    
    printf("Convolution Time: %.3g \n",tConv);
    printf("im2Col Time: %.3g \n",tIm2Col);
    printf("im2Col + Gemm Time: %.3g \n",tIm2ColGemm);
    printf("Implicit Gemm Time: %.3g \n",tImp);
    
    free(In);
    free(F);
    free(OutConv);
    free(OutI2c);
    free(OutImp);
    free(Aux);
}

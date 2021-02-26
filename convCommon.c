/**Common convolution code
 * 
 * This file contains the implementation of convolution related code used  by several sources in the project.
 * 
 * @author P. San Juan
 * @date 04/2020
 */

#include "convCommon.h"



/** Performs a multi-dimensional convolution using a naive algorithm.
 * 
 * Perfroms a multi-dimensional convolution applying the filter matrix F to the input tensor In.
 * 
 * @param[in] h input tensor hight
 * @param[in] w input tensor width
 * @param[in] c number of chanels of input tensor
 * @param[in] b batch Size
 * @param[in] In 1D-array containing a flattened version of the input tensor
 * @param[in] kh kernel height
 * @param[in] kw kernel width
 * @param[in] kn kernel number
 * @param[in] F 1-D array containing filter/kernel matrix
 * @param[out] Out 1-D array containing theresult of the convolution
 * @param[in] stride Stride to apply the krnels to the input tensor
 */
void convolutionNaive(const int h, const int w, const int c,const int b,const float* In,const int kh,const int kw, const int kn, const float* F, float* Out, const int stride)
{

    int ic, ikh, ikw, ih, iw, ib, ik,ho,wo, pad =0; //padding currently unsuported    
    float ZERO = 0.0;
    
    ho = floor((h - kh + 2 * pad) / stride + 1);
    wo = floor((w - kw + 2 * pad) / stride + 1);
    
    bli_ssetv(BLIS_NO_CONJUGATE,ho*wo*kn*b,&ZERO,Out,1);
    for(ib = 0;ib < b; ib++)    
        for( ic = 0; ic < c; ic++)
            for(iw = 0; iw < wo; iw++)                
                for(ih = 0; ih < ho; ih++)
                    for(ikw = 0; ikw < kw; ikw++)
                        for(ikh = 0; ikh < kh; ikh++)
                            for(ik=0;ik < kn; ik++)
                                Out[ik + ( ib*wo*ho + iw*ho + ih) *kn] += 
                                In[ib * c*h*w + ic * h*w + (iw * stride + ikw) * h + (stride * ih + ikh)] 
                                * F[ik + (ic * kh*kw + ikw * kh + ikh  ) * kn];

}


/** Performs a im2col transformation to the input tensor.
 * 
 * Applys the im2Col tranform to the input tensor. The im2col transform is used to perform a convolution using the GEMM kernel.
 * 
 * @param[in] h input tensor hight
 * @param[in] w input tensor width
 * @param[in] c number of chanels of input tensor
 * @param[in] b batch Size
 * @param[in] In 1D-array containing a flattened version of the input tensor
 * @param[in] kh kernel height
 * @param[in] kw kernel width
 * @param[in] stride Stride to apply the krnels to the input tensor
 * @param[out] Out Matrix (column major stored) containing the expanded matrix
 */
void im2Col(const int h, const int w, const int c, const int b,const float* In, const int kh, const int kw, const int stride,float* Out)
{
    int ic, ikh, ikw, ih, iw, ib,
        row,col, ho,wo,pad =0; //padding currently unsuported
   
    ho = floor((h - kh + 2 * pad) / stride + 1);
    wo = floor((w - kw + 2 * pad) / stride + 1);
   
    unsigned int cSize = h*w, //chanel memory leap in input tensor
                 coSize = ho*wo, //chanel memory leap in output matix
                 kSize = kh*kw, //kernel memory leap (single chanel)
                 bSize = c*h*w, //batch memory leap
                 ckSize = c * kSize, //kernels memory leap (all channels)
                 posib,
                 posic,
                 posiw,
                 posih,
                 posikw,
                 rowic,
                 rowikw,
                 colib,
                 coliw;
    
                 

    
    for(ib = 0;ib < b; ib++)    
    {
        colib = ib * coSize;
        posib = ib * bSize;
        #pragma omp parallel for private (iw,ih,ikw,ikh,row,col,rowic,posic,coliw,posiw,posih,rowikw,posikw) 
        for( ic = 0; ic < c; ic++)
        {
            rowic = ic *kSize;
            posic = ic * cSize + posib;
            for(iw = 0; iw < wo; iw++)   
            {
                coliw = colib + iw * ho;
                posiw = iw * stride * h + posic;
                for(ih = 0; ih < ho; ih++)
                {
                     //OPT col = ib * coSize + iw * ho + ih;
                    col = coliw + ih;
                    posih = stride * ih;
                    for(ikw = 0; ikw < kw; ikw++)
                    {
                        rowikw = rowic + ikw * kh;
                        posikw = posiw + ikw * h;
                        for(ikh = 0; ikh < kh; ikh++)
                        {
                             //OPT row = ic *kSize + ikw * kh + ikh; 
                            row = rowikw + ikh;
                            //printf("Writing into Out[%d,%d] from In[%d,%d,%d,%d]\n",
                              //    row,col,ib,ic, (iw * stride + ikw),(stride * ih + ikh));
                            //OPT Out[row + col * ckSize] = In[ib * bSize + ic * cSize + (iw * stride + ikw) * h + (stride * ih + ikh)];
                            //OPT Out[row + col * ckSize] = In[posib + posic + posikw + posih + ikh];
                            Out[row + col * ckSize] = In[posikw + posih + ikh];

                        }
                    }
                }   
            }
        }
    }
}


/**
 *  Performs a col2Im transformation to the input tensor.
 * 
 * Applys the col2Im tranform to an expanded matrix. The col2Im transform is used to 
 * return to the image space after the GEMM kernel in the backward stage of CNNs.
 * 
 * @param[in] h Image tensor hight
 * @param[in] w Image tensor width
 * @param[in] c number of chanels of image tensor
 * @param[in] b batch Size
* @param[in] mat Matrix (column major stored) containing the expanded matrix
 * @param[in] kh kernel height
 * @param[in] kw kernel width
 * @param[in] stride Stride to apply the krnels to the Image tensor
 * @param[out] Im 1D-array containing a flattened version of the Image tensor
 */
void col2Im(const int h, const int w, const int c, const int b,const float* mat, const int kh, const int kw, const int stride, const int pad, float* Im)
{
    int ic, ikh, ikw, ih, iw, ib,
        row,col, ho,wo, ldh;
   
    ho = floor((h - kh + 2 * pad) / stride + 1);
    wo = floor((w - kw + 2 * pad) / stride + 1);
    ldh = h + 2* pad; 
    
    unsigned int cSize = ldh * (w + 2 * pad), //chanel memory leap in output tensor
                 coSize = ho * wo, //chanel memory leap in input matix
                 kSize = kh * kw, //kernel memory leap (single chanel)
                 bSize = c * cSize, //batch memory leap
                 ckSize = c * kSize, //kernels memory leap (all channels)
                 posib, posic,
                 posiw, posih, posikw,
                 rowic, rowikw,
                 colib, coliw;
    
                 

    
    for(ib = 0;ib < b; ib++)    
    {
        colib = ib * coSize;
        posib = ib * bSize;
        #pragma omp parallel for private (iw,ih,ikw,ikh,row,col,rowic,posic,coliw,posiw,posih,rowikw,posikw) 
        for( ic = 0; ic < c; ic++)
        {
            rowic = ic *kSize;
            posic = ic * cSize + posib;
            for(iw = 0; iw < wo; iw++)   
            {
                coliw = colib + iw * ho;
                posiw = iw * stride * ldh + posic;
                for(ih = 0; ih < ho; ih++)
                {
                     //OPT col = ib * coSize + iw * ho + ih;
                    col = coliw + ih;
                    posih = stride * ih;
                    for(ikw = 0; ikw < kw; ikw++)
                    {
                        rowikw = rowic + ikw * kh;
                        posikw = posiw + ikw * ldh;
                        for(ikh = 0; ikh < kh; ikh++)
                        {
                             //OPT row = ic *kSize + ikw * kh + ikh; 
                            row = rowikw + ikh;
                            //OPT Im[ib * bSize + ic * cSize + (iw * stride + ikw) * ldh + (stride * ih + ikh)] += mat[row + col * ckSize] ;
                            //OPT Im[posib + posic + posikw + posih + ikh] += mat[row + col * ckSize];
                            Im[posikw + posih + ikh] += mat[row + col * ckSize];

                        }
                    }
                }   
            }
        }
    }
}


/** Adds padding to a stack of matrices
 * 
 * Adds padding in the X and Y axys to a 4-dymensional input tensor. Ther padding is aplied at rigth, left, top and bottom 
 * of the matrix formed by the two innermost dimensions.
 * 
 * @param[in] m input tensor heigth
 * @param[in] n input tensor width
 * @param[in] c number of chanels of input tensor
 * @param[in] b batch Size
 * @param[in] In 1D-array containing a flattened version of the input tensor
 * @param[in] pad Number of "pixels" to padd
 * @param[out] padM Output tensor with all its matrices padded
 */
void padMatrix(const int m, const int n, const int c, const int b, const float* In , const int pad, float * padM)
{
    
}


#include "convCommon.h"




void convolutionNaive(const int h, const int w, const int c,const int b,const float* In,const int kh,const int kw, const int kn, const float* F, float* Out, const int stride)
{

    int ic, ikh, ikw, ih, iw, ib, ik;    
    float ZERO = 0.0;
    
    
    bli_ssetv(BLIS_NO_CONJUGATE,h*w*kn*b,&ZERO,Out,1);
    for(ib = 0;ib < b; ib++)    
        for( ic = 0; ic < c; ic++)
            for(iw = 0; iw < w; iw++)                
                for(ih = 0; ih < h; ih++)
                    for(ikw = 0; ikw < kw; ikw++)
                        for(ikh = 0; ikh < kh; ikh++)
                            for(ik=0;ik < kn; ik++)
                                Out[ik + ( ib*w*h + iw*h + ih) *kn] += 
                                In[ib * c*h*w + ic * h*w + (iw * stride + ikw) * h + (stride * ih + ikh)] 
                                * F[ik + (ic * kh*kw + ikw * kh + ikh  ) * kn];

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

void padMatrix()
{
    
}
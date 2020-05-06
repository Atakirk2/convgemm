
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
    int ic, ikh, ikw, ih, iw, ib,
        row,col;
   
    unsigned int cSize = h*w, //chanel memory leap
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
        colib = ib * cSize;
        posib = ib * bSize;
        #pragma omp parallel for private (iw,ih,ikw,ikh,row,col,rowic,posic,coliw,posiw,posih,rowikw,posikw) 
        for( ic = 0; ic < c; ic++)
        {
            rowic = ic *kSize;
            posic = ic * cSize + posib;
            for(iw = 0; iw < w; iw++)   
            {
                coliw = colib + iw * h;
                posiw = iw * stride * h + posic;
                for(ih = 0; ih < h; ih++)
                {
                     //OPT col = ib * cSize + iw * h + ih;
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

void padMatrix()
{
    
}

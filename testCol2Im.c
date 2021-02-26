/**Convolution tester
 * 
 * This test compares the naive convolution  with the imtocol + gemm and the convGemm approaches.
 * The test is performed for ARMCortex  A-57
 * 
 * @author P. San Juan
 * @date 01/2021
 */

#include "convCommon.h"
#include "convGemm.h"

void sset0sM(unsigned int m, unsigned int n, float *restrict M,
             unsigned int ldm);

int print_matrices( int m, int n, char *name, float *M, int ldm,  char *name2, float *M2, int ldm2)
{

  int i, j;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      printf( "   (%d,%d) = %s[%22.15e]    [%22.15e]%s;\n",  i, j,name, (double)M[i +j * ldm], (double)M2[i +j * ldm2],name2);

  return 0;
}


/** Compares two matrices and returns the difference between them.
 * 
 * @param m Rows of matrices.
 * @param n Cols of matrices.
 * @param M First matrix.
 * @param ldm Leading dimension of M. 
 * @param M2 Second matrix.
 * @param ldm2 Leading dimension of M2.
 * @return  |N-M2|_F/|M2|_F
 * */
float compareMatrix(const int m,const int n, float* M, const int ldm, float* M2, const int ldm2){
    

    float norm = 0, normM2 = 0;
    
    bli_snormfv(m*n,M2,1,&normM2);
    if(normM2 == 0)
        printf("WARN: norm(M2)=0\n");
    bli_ssubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,M,1,ldm,M2,1,ldm2);
    bli_snormfv(m*n,M2,1,&norm);
    
    return norm/normM2;
}

void extractMatrix(unsigned int m, unsigned int n, unsigned int pad, float * Mpad, float* M )
{
    int i, j, ip, jp;
    
    for(j = 0, jp = pad; j < m; j++, jp++)
        for(i = 0,ip = pad; i < n; i++,ip++)
            M[i + j * n] = Mpad[ip + jp * (n +2 * pad)];
    
    
}

void extractImages(unsigned int h, unsigned int w, unsigned int c, unsigned int b, unsigned int pad, float * Mpad, float* M )
{
    int ih, iw, ic, ib, ihp, iwp;
    
    unsigned int cSize = h * w;
    unsigned int bSize = cSize * c;
    
    unsigned int cSizePad = (h + 2 * pad) * (w + 2 * pad);
    unsigned int bSizePad = cSizePad * c;
    
    for(ib = 0; ib < b; ib++)
        for(ic = 0; ic < c; ic++)
            for(iw = 0, iwp = pad; iw < w; iw++, iwp++)
                for(ih = 0,ihp = pad; ih < h; ih++,ihp++)
                    M[ih + iw * h + ic * (cSize) + ib * bSize] = 
                        Mpad[ihp + iwp * (h +2 * pad) + ic * cSizePad + ib * bSizePad];
}



int main( int argc, char** argv )
{
    double tCol2Im = 0.0, tCol2ImGemm = 0, tImp = 0.0, tIni,
            perfPeak,perfGemm, perfImp;
            
    float ONE = 1, ZERO = 0;
    
    int i, m, n, 
        repe, //repetitions
        iZERO = 0;
        
     int h,w,c,b, //input dims
         ho,wo, //oputput dimms
         kh,kw, kn, //kernel dimms
         stride,pad; //algorithm parameters
        
    
    float *F, *dX, *dXImp, * dXextract, 
         *dY,
         *Aux,
         *Ac_pack, *Bc_pack, *Cc_pack;

         
    
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
                
    printf("Allocating matrices...\n");
    
  
    ho = floor((h - kh + 2 * pad) / stride + 1);
    wo = floor((w - kw + 2 * pad) / stride + 1);
    
    //input matrices
    dY = (float*) malloc(ho*wo*kn*b * sizeof(float));
    F = (float*) malloc(kh*kw*c *kn * sizeof(float));

    //auxiliar matrices
    Ac_pack = (float*) aligned_alloc(4096,BLOCK_MC*BLOCK_KC*sizeof(float));
    Bc_pack = (float*) aligned_alloc(4096,BLOCK_KC*BLOCK_NC*sizeof(float));
    Cc_pack = (float*) aligned_alloc(4096,kh*kw*c*BLOCK_NC*sizeof(float));
    Aux = (float*) malloc(c*kh*kw * ho*wo*b * sizeof(float));
    
    //output matrices 
    dX = (float*) malloc((h+2 * pad)*(w + 2 * pad)*c *b * sizeof(float));
#ifdef cIsPadded
    dXImp = (float*) malloc((h + 2 * pad)*(w+ 2 * pad)*c *b * sizeof(float));
#else
    dXImp = (float*) malloc(h *w*c *b * sizeof(float));
    dXextract = (float*) malloc(h *w*c *b * sizeof(float));
#endif
    
    printf("Generating random matrices...\n");
    bli_srandm( 0, BLIS_DENSE, kn, ho*wo*b, dY, 1, kn );
    bli_srandm( 0, BLIS_DENSE, kn, kh*kw*c, F, 1, kn );
    
    printf("Starting evaluation...\n");
    for(i = 0; i <  repe; i++)
    {
        //Timing gemm + im2col
      tIni = bli_clock();
        bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,kh*kw*c,ho*wo*b,kn,&ONE,F,1,kh*kw*c,dY,1,kn,&ZERO,Aux,1,kh*kw*c);
      tCol2ImGemm += bli_clock() -tIni;
        sset0sM((h + 2 * pad) * (w + 2 * pad) * c * b,1, dX, 0);
        col2Im(h,w,c,b,Aux,kh,kw, stride, pad,dX);
        
      tCol2Im += bli_clock() -tIni;
        
        //Timing implicint gemm
        tIni = bli_clock();
        sconvGemm_back(kh,kw,c,kn,1,F, h,w,b, stride, stride, pad,pad,dY,dXImp,Ac_pack,Bc_pack,Cc_pack);
        tImp += bli_clock() -tIni;

    }
    
    
    tCol2Im/=repe;
    tCol2ImGemm/=repe;
    tImp/=repe;
    
    perfImp = ( 2.0 * kn*ho*wo*b*kh*kw*c ) / ( tImp * 1.0e9 );
    perfGemm = ( 2.0 * kn*ho*wo*b*kh*kw*c ) / ( (tCol2ImGemm) * 1.0e9 );
    perfPeak = 2.035 * 8 ;// cpuFreq * flopscycle
    
    printf("Gemm + col2Im Time: %.4g [Gemm: %.4g , Col2Im: %.4g] GFLOPS(gemm): %.5g \n",tCol2Im, tCol2ImGemm, tCol2Im - tCol2ImGemm,perfGemm);
    printf("Implicit Gemm Time: %.4g GFLOPS: %.5g\n",tImp,perfImp);
    printf("Peak performance: %g, im2col gemm: %g, implicit gemm: %g\n",perfPeak,perfGemm/perfPeak,perfImp/perfPeak);
    
#ifdef cIsPadded
    //print_matrices( (h + 2 * pad)*(w+ 2 * pad)*c *b , 1, "dX", dX, 0,  "dXImp", dXImp, 0);
    printf("norm(dX-dXImp)=%g\n",compareMatrix((h + 2 * pad) * (w + 2 * pad) * c * b,1,dX,0,dXImp,0));
#else    
    extractImages(h,w,c,b, pad, dX, dXextract);
    //print_matrices( h *w*c *b , 1, "dX", dXextract, 0,  "dXImp", dXImp, 0);
    printf("norm(dX-dXImp)=%g\n",compareMatrix(h * w * c * b,1,dXextract,0,dXImp,0));
#endif
    

    
    
    free(dX);
    free(F);
    free(dY);
    free(dXImp);
    free(Aux);
    free(Ac_pack);
    free(Bc_pack);
    free(Cc_pack);
}

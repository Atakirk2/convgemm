/**Convolution tester
 * 
 * This test compares the naive convolution  with the imtocol + gemm and the convGemm approaches.
 * The test is performed for ARMCortex  A-57
 * 
 * @author P. San Juan
 * @date 04/2020
 */

#include "convCommon.h"
#include "convGemm.h"


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
    bli_ssubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,M,1,ldm,M2,1,ldm2);
    bli_snormfv(m*n,M2,1,&norm);
    
    return norm/normM2;
}

int main( int argc, char** argv )
{
    double tConv = 0.0, tIm2Col = 0.0, tIm2ColGemm = 0, tImp = 0.0, tIni,
            perfPeak,perfGemm, perfImp;
            
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
         *Aux,
         *Ac_pack, *Bc_pack;

         
    
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
    
    //input matrices
    In = (float*) malloc(h*w*c *b * sizeof(float));
    F = (float*) malloc(kh*kw*c *kn * sizeof(float));

    ho = floor((h - kh + 2 * pad) / stride + 1);
    wo = floor((w - kw + 2 * pad) / stride + 1);
    
    //auxiliar matrices
    Ac_pack = (float*) aligned_alloc(4096,BLOCK_MC*BLOCK_KC*sizeof(float));
    Bc_pack = (float*) aligned_alloc(4096,BLOCK_KC*BLOCK_NC*sizeof(float));
    Aux = (float*) malloc(c*kh*kw * ho*wo*b * sizeof(float));
    
    //output matrices 
    OutConv = (float*) malloc(ho*wo*kn*b * sizeof(float));
    OutI2c= (float*) malloc(ho*wo*kn*b * sizeof(float));
    OutImp = (float*) malloc(ho*wo*kn*b * sizeof(float));
    
    
    printf("Generating random matrices...\n");
    bli_srandm( 0, BLIS_DENSE, b, h*w*c, In, 1, b );
    bli_srandm( 0, BLIS_DENSE, kn, kh*kw*c, F, 1, kn );
    
    printf("Starting evaluation...\n");
    for(i = 0; i <  repe; i++)
    {
        //Timing naive convolution
        tIni = bli_clock();
        convolutionNaive(h,w,c,b,In,kh,kw,kn,F,OutConv, stride);
        tConv += bli_clock() - tIni;

        //Timing im2col +gemm
        tIni = bli_clock();
        im2Col (h,w,c,b,In,kh,kw, stride,Aux);
        tIm2Col += bli_clock() -tIni;
        bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,kn,ho*wo*b,kh*kw*c,&ONE,F,1,kn,Aux,1,kh*kw*c,&ZERO,OutI2c,1,kn);
        //sgemm_cust(kn,ho*wo*b,kh*kw*c,1,F,kn,Aux,kh*kw*c,0,OutI2c,kn,Ac_pack,Bc_pack);
        tIm2ColGemm += bli_clock() -tIni;
        
        
        
        //Timing implicint gemm
        tIni = bli_clock();
        sconvGemm(kh,kw,c,kn,1,F, h,w,b, stride, stride, In, 0,OutImp,Ac_pack,Bc_pack);
        tImp += bli_clock() -tIni;

    }
    
    tConv /=repe;
    tIm2Col/=repe;
    tIm2ColGemm/=repe;
    tImp/=repe;
    
    perfImp = ( 2.0 * kn*ho*wo*b*kh*kw*c ) / ( tImp * 1.0e9 );
    perfGemm = ( 2.0 * kn*ho*wo*b*kh*kw*c ) / ( (tIm2ColGemm-tIm2Col) * 1.0e9 );
    perfPeak = 2.035 * 8 ;// cpuFreq * flopscycle
    
    printf("Convolution Time: %.4g \n",tConv);
    printf("im2Col + Gemm Time: %.4g [Im2Col: %.4g , Gemm: %.4g] GFLOPS(gemm): %.5g \n",tIm2ColGemm, tIm2Col, tIm2ColGemm -tIm2Col,perfGemm);
    printf("Implicit Gemm Time: %.4g GFLOPS: %.5g\n",tImp,perfImp);
    printf("Peak performance: %g, im2col gemm: %g, implicit gemm: %g\n",perfPeak,perfGemm/perfPeak,perfImp/perfPeak);
    
    
    printf("norm(OutIm2col-OutImplicit)=%g\n",compareMatrix(kn,ho*wo*b,OutI2c,kn,OutImp,kn));
    printf("norm(OutIm2col-OutNaive)=%g\n",compareMatrix(kn,ho*wo*b,OutI2c,kn,OutConv,kn));
    
    free(In);
    free(F);
    free(OutConv);
    free(OutI2c);
    free(OutImp);
    free(Aux);
    free(Ac_pack);
    free(Bc_pack);
}

/**Gemm block size evaluator
 * 
 * This test performs a block size evaluation for GEMM
 * The test is performed for NVIDIA Carmel 
 * 
 * @author P. San Juan
 * @date 04/2020
 */


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemmConv.h"
#include "blis.h"

#ifdef fp_D
    #define fpType double
    #define EPS 1e-16
#elif fp_H
    #define fpType _Float16
    #define EPS 5e-4
#elif i_16
    #define fp_H 1
    #define fpType int16_t
    #define EPS 5e-4
#else
    #define fpType float
    #define EPS 6e-8
#endif

void irandm(unsigned int m, unsigned int n, fpType *M,unsigned int ldm);


int main( int argc, char** argv )
{
    double tOwn, tIni, gflopsOwn;
            
    fpType ONE = 1, ZERO = 0;
    
    int i,
        m, n, k, //matrix dimms
        mc, nc, kc,
        minMc,minKc,stepMc,stepKc,
        repe;
    
    fpType *A, *B, *C,
            *Ac_pack, *Bc_pack;
#ifdef fp_H
    float* Afloat, *Bfloat; 
#endif

    fpType norm = 0, normOrig = 0;
    
    if (argc != 8)
    {
        printf("Block size evaluation for GEMM.\n");
        printf("\tm,n,k: Matrix product dimensions.\n");
        printf("\trepe: number of repetitions of the test.\n");
        printf("Usage: %s <m> <n> <k> <mc> <nc> <kc> <repe>\n", argv[0]);
        return -1;
    }

    m =atoi(argv[1]);  
    n =atoi(argv[2]);
    k = atoi(argv[3]);
    mc =atoi(argv[4]);  
    nc =atoi(argv[5]);
    kc = atoi(argv[6]);
    repe =atoi(argv[7]);
    
    minKc = 16;
    stepKc = 16;
    minMc = 16;
    stepMc = 16;
   
    
#ifdef fp_D
    dBLOCK_NC = nc; 
#elif fp_H
    hBLOCK_NC = nc; 
#else
    BLOCK_NC = nc;
#endif
    
    
    //Alloca matrices
    A = (fpType*) malloc(m*k * sizeof(fpType));
    B = (fpType*) malloc(k*n * sizeof(fpType));
    C = (fpType*) malloc(m*n * sizeof(fpType));
    
    Ac_pack = (fpType*) aligned_alloc(4096,mc*kc*sizeof(fpType));
    Bc_pack = (fpType*) aligned_alloc(4096,kc*nc*sizeof(fpType));
    
#ifdef fp_H
    Afloat = (float*) malloc(m*k * sizeof(float));
    Bfloat = (float*) malloc(k*n * sizeof(float));
#endif

#ifdef fp_D
    bli_drandm( 0, BLIS_DENSE, m, k, A, 1, m );
    bli_drandm( 0, BLIS_DENSE, k, n, B, 1, k );
#elif i_16
    irandm( m, k, A, m );
    irandm( k, n, B, k );
#elif fp_H
    bli_srandm( 0, BLIS_DENSE, m, k, Afloat, 1, m );
    decreasePrecissionV_SH(m*k,Afloat,A);
    bli_srandm( 0, BLIS_DENSE, k, n, Bfloat, 1, k );
    decreasePrecissionV_SH(k*n,Bfloat,B);    
#else
    bli_srandm( 0, BLIS_DENSE, m, k, A, 1, m );
    bli_srandm( 0, BLIS_DENSE, k, n, B, 1, k );
#endif
    
#ifdef fp_D
        printf("Precision: double\n");
#elif i_16
        printf("Precision: int 16\n");
#elif fp_H
        printf("Precision: half\n");
#else
        printf("Precision: simple\n");
#endif
    
    //Timing  gemm 
#ifdef fp_D
    for(dBLOCK_MC=minMc;dBLOCK_MC <= mc; dBLOCK_MC+=stepMc)
        for(dBLOCK_KC=minKc;dBLOCK_KC <= kc; dBLOCK_KC+=stepKc) 
#elif fp_H
    for(hBLOCK_MC=minMc;hBLOCK_MC <= mc; hBLOCK_MC+=stepMc)
        for(hBLOCK_KC=minKc;hBLOCK_KC <= kc; hBLOCK_KC+=stepKc) 
#else
    for(BLOCK_MC=minMc;BLOCK_MC <= mc; BLOCK_MC+=stepMc)
        for(BLOCK_KC=minKc;BLOCK_KC <= kc; BLOCK_KC+=stepKc)
#endif
        {
            tIni = bli_clock();
            for(i = 0; i <  repe; i++)
            {
            #ifdef fp_D
                dgemm_cust(m,n,k,1.0,A,m,B,k,0.0,C,m,Ac_pack,Bc_pack);
            #elif i_16
                i16gemm_cust(m,n,k,1,A,m,B,k,0,C,m,Ac_pack,Bc_pack);
            #elif fp_H
                hgemm_cust(m,n,k,1.0,A,m,B,k,0.0,C,m,Ac_pack,Bc_pack);
            #else
                sgemm_cust(m,n,k,1.0,A,m,B,k,0.0,C,m,Ac_pack,Bc_pack);
            #endif
            }
            tOwn = bli_clock() -tIni;
            tOwn/=repe;

            gflopsOwn = ( 2.0 * m * k * n ) / ( tOwn * 1.0e9 );

            #ifdef fp_D
                printf("MC=%d, KC=%d ,NC=%d, T[%.3f], P[%.3f]\n",dBLOCK_MC,dBLOCK_KC,dBLOCK_NC, tOwn,gflopsOwn);
            #elif fp_H
                printf("MC=%d, KC=%d ,NC=%d, T[%.3f], P[%.3f]\n",hBLOCK_MC,hBLOCK_KC,hBLOCK_NC, tOwn,gflopsOwn);
            #else
                printf("MC=%d, KC=%d ,NC=%d, T[%.3f], P[%.3f]\n",BLOCK_MC,BLOCK_KC,BLOCK_NC, tOwn,gflopsOwn);
            #endif
            
        }
    
    free(A);
    free(B);
    free(C);
    
}

void irandm(unsigned int m, unsigned int n, fpType *M,unsigned int ldm)
{
    int i,j;
    
    for ( j=0; j<n; j++ )
        for ( i=0; i<m; i++ )
            M[i+j*ldm] = rand() % 32767;
}

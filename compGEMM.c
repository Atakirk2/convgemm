/**GEMM comparator
 * 
 * This test performs a comparison between BLIS GEMM and the custom GEMM.
 * The test is performed for ARMCortex  A-57
 * 
 * @author P. San Juan
 * @date 04/2020
 */


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemmConv.h"
#include "blis.h"



#define EPS 1e10



int main( int argc, char** argv )
{
    double tBlis, tOwn, tIni,
            gflopsBlis, gflopsOwn;
            
    float ONE = 1, ZERO = 0;
    
    int i,
        m, n, k, //matrix dimms
        repe;
    
    float *A, *B, *CBlis, *COwn,
            *Ac_pack, *Bc_pack;

    float norm = 0, normOrig = 0;
    
    if (argc != 5)
    {
        printf("Comparison of blis default GEMM and custom GEMM.\n");
        printf("\tm,n,k: Matrix product dimensions.\n");
        printf("\trepe: number of repetitions of the test.\n");
        printf("Usage: %s <m> <n> <k> <repe>\n", argv[0]);
        return -1;
    }

    m =atoi(argv[1]);  
    n =atoi(argv[2]);
    k = atoi(argv[3]);
    repe  =atoi(argv[4]);
    
    A = (float*) malloc(m*k * sizeof(float));
    B = (float*) malloc(k*n * sizeof(float));
    CBlis = (float*) malloc(m*n * sizeof(float));
    COwn = (float*) malloc(m*n * sizeof(float));
    
    Ac_pack = (float*) aligned_alloc(4096,BLOCK_MC*BLOCK_KC*sizeof(float));
    Bc_pack = (float*) aligned_alloc(4096,BLOCK_KC*BLOCK_NC*sizeof(float));
    
    
    bli_srandm( 0, BLIS_DENSE, m, k, A, 1, m );
    bli_srandm( 0, BLIS_DENSE, k, n, B, 1, k );

    for(i = 0; i <  repe; i++)
    {
        //Timing gemm blis
        tIni = bli_clock();
        bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&ONE,A,1,m,B,1,k,&ZERO,CBlis,1,m);
        tBlis += bli_clock() - tIni;
        
        //Timing custom gemm 
        tIni = bli_clock();
        sgemm_cust(m,n,k,1.0,A,m,B,k,0.0,COwn,m,Ac_pack,Bc_pack);
        tOwn += bli_clock() -tIni;
        
                
#ifdef COMPARE
         bli_snormfv(m*n,COwn,1,&normOrig);
         bli_ssubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,CBlis,1,m,COwn,1,m);
         bli_snormfv(m*n,COwn,1,&norm);
         printf("Approximation error: %g\n",norm/normOrig);
         if (norm > EPS)
         {
             printf("Error threshold exceeded\n");
             exit(1);
         }
         
#endif
    }
    
    tBlis /=repe;
    tOwn/=repe;
    
    gflopsBlis = ( 2.0 * m * k * n ) / ( tBlis * 1.0e9 );
    gflopsOwn = ( 2.0 * m * k * n ) / ( tOwn * 1.0e9 );
    
    printf("BLIS Time: %.3f GFlops: %.3f\n",tBlis,gflopsBlis);
    printf("Custom Time: %.3f GFlops: %.3f\n",tOwn,gflopsOwn);
    
    
    free(A);
    free(B);
    free(CBlis);
    free(COwn);
    
}


/**
 * Print a matrix to standard output.
 * 
 * @param[in] name Label for matrix name
 * @param[in] m Row dimension
 * @param[in] n Column dimension
 * @param[in] A Matrix
 * @param[in] Alda Leading dimension
 *
 */
int print_matrix( char *name, int m, int n, float *A, int Alda )
{

  int i, j;

  for ( j=1; j<=n; j++ )
    for ( i=1; i<=m; i++ )
      printf( "   %s(%d,%d) = %22.15e;\n", name, i, j, A[m*(j-1)+(i-1)]);

  return 0;
}


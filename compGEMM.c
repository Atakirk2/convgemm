#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemmConv.h"
#include "blis.h"



#define EPS 1e10




int main( int argc, char** argv )
{
    double tBlis, tOwn, tIni,
            gflopsBlis, gflopsOwn,
            ONE = 1;
    
    int i,
        m, n, k, //matrix dimms
        repe,
        iZERO = 0;
    
    double *A, *B, *CBlis, *COwn;

    double norm = 0;
    
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
    
    A = (double*) malloc(m*k * sizeof(double));
    B = (double*) malloc(k*n * sizeof(double));
    CBlis = (double*) malloc(m*n * sizeof(double));
    COwn = (double*) malloc(m*n * sizeof(double));

    for(i = 0; i <  repe; i++)
    {
        //Timing gemm blis
        tIni = bli_clock();
        bli_dgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&ONE,A,1,m,B,1,k,&ONE,CBlis,1,m);
        tBlis += bli_clock() - tIni;
        
        //Timing custom gemm 
        tIni = bli_clock();
        dgemm_conv(m,n,k,1,A,m,B,k,1,COwn,m);
        tOwn += bli_clock() -tIni;
        
#ifdef COMPARE
         bli_dsubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,CBlis,1,m,COwn,1,m);
         bli_dnormfv(m*n,COwn,1,&norm);
         printf("Approximation error: %g\n",norm);
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

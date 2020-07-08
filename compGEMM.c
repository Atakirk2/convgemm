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





#ifdef fp_D
    #define fpType double
    #define EPS 1e-16
#elif fp_H
    #define fpType _Float16
    #define EPS 5e-4
#elif fp_HS
    #define fpType _Float16
    #define fp_H 1
    #define EPS 5e-4
#else
    #define fpType float
    #define EPS 6e-8
#endif

double compareMatrix(int m, int n, fpType *M, int ldm, fpType *M2, int ldm2 );
int print_matrix( char *name, int m, int n, fpType *M, int ldm );
int print_matrices( int m, int n, char *name, fpType *M, int ldm,  char *name2, fpType *M2, int ldm2);

int main( int argc, char** argv )
{
    double tBlis, tOwn, tIni,
            gflopsBlis, gflopsOwn;
            
    fpType ONE = 1, ZERO = 0;
    
    int i,
        m, n, k, //matrix dimms
        repe;
    
    fpType *A, *B, *CBlis, *COwn,
            *Ac_pack, *Bc_pack;
#ifdef fp_H
    float* Afloat, *Bfloat, *Cfloat; 
    float fONE = 1, fZERO = 0;
#endif

    fpType norm = 0, normOrig = 0;
    
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
    
    A = (fpType*) malloc(m*k * sizeof(fpType));
    B = (fpType*) malloc(k*n * sizeof(fpType));
    CBlis = (fpType*) malloc(m*n * sizeof(fpType));
    COwn = (fpType*) malloc(m*n * sizeof(fpType));
        //print_matrix("CBuff",m,n,COwn,m);
        //print_matrix("CBuff1",m,n,COwn,m);
#ifdef fp_H
    Afloat = (float*) malloc(m*k * sizeof(float));
    Bfloat = (float*) malloc(k*n * sizeof(float));
    Cfloat = (float*) malloc(m*n * sizeof(float));
#endif
    
    Ac_pack = (fpType*) aligned_alloc(4096,BLOCK_MC*BLOCK_KC*sizeof(fpType));
    Bc_pack = (fpType*) aligned_alloc(4096,BLOCK_KC*BLOCK_NC*sizeof(fpType));
    

#ifdef fp_D
    bli_drandm( 0, BLIS_DENSE, m, k, A, 1, m );
    bli_drandm( 0, BLIS_DENSE, k, n, B, 1, k );
#elif fp_H
   //print_matrix("CBuff",m,n,COwn,m);
    bli_srandm( 0, BLIS_DENSE, m, k, Afloat, 1, m );
    decreasePrecissionV_SH(m*k,Afloat,A);
    bli_srandm( 0, BLIS_DENSE, k, n, Bfloat, 1, k );
    decreasePrecissionV_SH(k*n,Bfloat,B);    
#else
    bli_srandm( 0, BLIS_DENSE, m, k, A, 1, m );
    bli_srandm( 0, BLIS_DENSE, k, n, B, 1, k );
#endif


    //Timing gemm blis
    tIni = bli_clock();
    for(i = 0; i <  repe; i++)
    {
#ifdef fp_D
        bli_dgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&ONE,A,1,m,B,1,k,&ZERO,CBlis,1,m);
#elif fp_H
        increasePrecissionV_HS(m*k,A,Afloat);
        increasePrecissionV_HS(k*n,B,Bfloat);
        bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&fONE,Afloat,1,m,Bfloat,1,k,&fZERO,Cfloat,1,m); 
        decreasePrecissionV_SH(m*n,Cfloat,CBlis);
#else
        bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&ONE,A,1,m,B,1,k,&ZERO,CBlis,1,m);
#endif
    }
    tBlis = bli_clock() - tIni;
    
    
    //Timing custom gemm 
    tIni = bli_clock();
    for(i = 0; i <  repe; i++)
    {
#ifdef fp_D
        dgemm_cust(m,n,k,1.0,A,m,B,k,0.0,COwn,m,Ac_pack,Bc_pack);
#elif fp_HS
        hsgemm_cust(m,n,k,1.0,A,m,B,k,0.0,COwn,m,Ac_pack,Bc_pack, Cfloat);
#elif fp_H
        hgemm_cust(m,n,k,1.0,A,m,B,k,0.0,COwn,m,Ac_pack,Bc_pack);
#else
        sgemm_cust(m,n,k,1.0,A,m,B,k,0.0,COwn,m,Ac_pack,Bc_pack);
#endif
    }
    tOwn = bli_clock() -tIni;
        
                
#ifdef COMPARE
  #ifdef fp_D
         bli_dnormfv(m*n,COwn,1,&normOrig);
         bli_dsubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,CBlis,1,m,COwn,1,m);
         bli_dnormfv(m*n,COwn,1,&norm);
         printf("Approximation error: %g\n",norm/normOrig);
         if (norm > EPS)
         {
             printf("Error threshold exceeded\n");
             exit(1);
         }
  #elif fp_H
        norm = compareMatrix(m,n,COwn, m,CBlis,m);
        //print_matrices( m, n,  "Own", COwn, m, "Blis", CBlis, m);
      
        
        printf("Approximation error: %g\n",(double)norm);
         if (norm > EPS)
         {
             printf("Error threshold exceeded\n");
            // exit(1);
         }
  #else
         bli_snormfv(m*n,COwn,1,&normOrig);
         norm = compareMatrix(m,n,COwn, m,CBlis,m);
         printf("Approximation erro compareMatrix: %g\n",norm);
         bli_ssubm(0,BLIS_NONUNIT_DIAG,BLIS_DENSE,BLIS_NO_TRANSPOSE,m,n,CBlis,1,m,COwn,1,m);
         bli_snormfv(m*n,COwn,1,&norm);
         
         printf("Approximation error: %g\n",norm/normOrig);
         if (norm > EPS)
         {
             printf("Error threshold exceeded\n");
             exit(1);
         }
  #endif
#endif
    
    
    tBlis /=repe;
    tOwn/=repe;
    
    gflopsBlis = ( 2.0 * m * k * n ) / ( tBlis * 1.0e9 );
    gflopsOwn = ( 2.0 * m * k * n ) / ( tOwn * 1.0e9 );
    
#ifdef fp_D
        printf("Precision: double\n");
#elif fp_HS
        printf("Precision: half with simple precission accumulation\n");
#elif fp_H
        printf("Precision: half\n");
#else
        printf("Precision: simple\n");
#endif
        
    printf("BLIS Time: %.3f GFlops: %.3f\n",tBlis,gflopsBlis);
    printf("Custom Time: %.3f GFlops: %.3f\n",tOwn,gflopsOwn);
    
    
    free(A);
    free(B);
    free(CBlis);
    free(COwn);
    
}

double compareMatrix(int m, int n, fpType *M, int ldm, fpType *M2, int ldm2 )
{
    int i,j;
    double Aux = 0;
    double Aux2 = 0;
    
    
    for(j=0; j < n; j++)
        for(i=0; i < m; i++)
        {
            Aux += pow(M[i+j*ldm] - M2[i + j* ldm2],2);
            Aux2+= M[i+j*ldm] * M[i+j*ldm];
            
        }
    return sqrt(Aux)/sqrt(Aux2);
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
int print_matrix( char *name, int m, int n, fpType *M, int ldm )
{

  int i, j;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      printf( "   %s(%d,%d) = %22.15e;\n", name, i, j, (double)M[i +j * ldm]);

  return 0;
}

int print_matrices( int m, int n, char *name, fpType *M, int ldm,  char *name2, fpType *M2, int ldm2)
{

  int i, j;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      printf( "   (%d,%d) = %s[%22.15e]    [%22.15e]%s;\n",  i, j,name, (double)M[i +j * ldm], (double)M2[i +j * ldm2],name2);

  return 0;
}

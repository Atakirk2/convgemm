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
    #define B_MC dBLOCK_MC
    #define B_NC dBLOCK_NC
    #define B_KC dBLOCK_KC
#elif fp_H
    #define fpType _Float16
    #define EPS 5e-4
    #define B_MC hBLOCK_MC
    #define B_NC hBLOCK_NC
    #define B_KC hBLOCK_KC
#elif fp_HS
    #define fpType _Float16
    #define fp_H 1
    #define EPS 5e-4
    #define B_MC hBLOCK_MC
    #define B_NC hBLOCK_NC
    #define B_KC hBLOCK_KC
#else
    #define fpType float
    #define EPS 6e-8
    #define B_MC BLOCK_MC
    #define B_NC BLOCK_NC
    #define B_KC BLOCK_KC
#endif

double compareMatrix(int m, int n, fpType *M, int ldm, fpType *M2, int ldm2 );
int print_matrix( char *name, int m, int n, fpType *M, int ldm );
int print_matrices( int m, int n, char *name, fpType *M, int ldm,  char *name2, fpType *M2, int ldm2);
void naiveGemm(unsigned m, unsigned n, unsigned k, fpType alpha, fpType *A,unsigned lda,fpType *B, unsigned ldb, fpType beta, fpType *C, unsigned ldc ,fpType *Ac_pack, fpType *Bc_pack);

int main( int argc, char** argv )
{
    double tBlis, tOwn, tIni,
            gflopsBlis, gflopsOwn;
            
    char* precision;
    
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
    
    struct threadStruct thrSt;
    thrSt.JC = 1;
    thrSt.IC = 1;
    thrSt.JR = 1;
    thrSt.IR = 1;
    thrSt.PR = 1;
    
    if (argc != 5 && argc != 9)
    {
        printf("Comparison of blis default GEMM and custom GEMM.\n");
        printf("\tm,n,k: Matrix product dimensions.\n");
        printf("\trepe: number of repetitions of the test.\n");
        printf("Usage: %s <m> <n> <k> <repe> [<JC> <IC> <JR> <IR>]\n", argv[0]);
        return -1;
    }

    m = atoi(argv[1]);  
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    repe = atoi(argv[4]);
    if(argc == 9)
    {
        thrSt.JC = atoi(argv[5]);
        thrSt.IC = atoi(argv[6]);
        thrSt.JR = atoi(argv[7]);
        thrSt.IR = atoi(argv[8]);
    }
    
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
    
    Ac_pack = (fpType*) aligned_alloc(4096,B_MC*B_KC*sizeof(fpType));
    Bc_pack = (fpType*) aligned_alloc(4096,B_KC*B_NC*sizeof(fpType));
    

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
        //naiveGemm(m,n,k,1.0,A,m,B,k,0.0,CBlis,m,Ac_pack,Bc_pack);
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
        hgemm_cust(m,n,k,1.0,A,m,B,k,0.0,COwn,m,Ac_pack,Bc_pack, thrSt);
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
        //printf("Precision: double\n");
        precision = "double";
#elif fp_HS
        //printf("Precision: half with simple precission accumulation\n");
        precision = "half with simple precission accumulation";
#elif fp_H
        //printf("Precision: half\n");
        precision = "half";
#else
        //printf("Precision: simple\n");
        precision = "single";
#endif
        
   // printf("BLIS Time: %.3f GFlops: %.3f\n",tBlis,gflopsBlis);
    //printf("Custom Time: %.3f GFlops: %.3f\n",tOwn,gflopsOwn);
    
    printf("Prec[%s],Size[%d,%d,%d],BLIS[T=%.4f,P=%.3f],Custom[T=%.4f,P=%.3f]\n",precision,m,n,k,tBlis,gflopsBlis,tOwn,gflopsOwn);
    
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

/*Naive gemm for comparison purposes*/
void naiveGemm(unsigned m, unsigned n, unsigned k, fpType alpha, fpType *A,unsigned lda,fpType *B, unsigned ldb, fpType beta, fpType *C, unsigned ldc ,fpType *Ac_pack, fpType *Bc_pack)
{
    int i,j,z;
    fpType *AB = (fpType*) malloc(m*n * sizeof(fpType));
    

    for(j = 0; j < n; j++)
        for (z = 0; z < k; z++)
            for (i = 0; i < m; i++)
                AB[i + j * ldc] += A[i + z * lda] * B[ z + j * ldb];
            
    for(j = 0; j < n; j++)
        for (i = 0; i < m; i++)
            C[i + j* ldc] = AB[i + j * ldc] + beta * C[i+j*ldc];
    
}

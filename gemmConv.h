/**GEMM and GEMM_conv codef
 * 
 * This file contains the declaration of all functions and constants related with the GEMM computation.
 * 
 * @author P. San Juan
 * @date 04/2020
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <stdint.h>


#ifdef runtimeBLCKS
    #define dBLOCK_NR 8
    #define dBLOCK_MR 6
    int dBLOCK_MC,
        dBLOCK_NC,
        dBLOCK_KC;
    #define hBLOCK_NR 8
    #define hBLOCK_MR 24
    int hBLOCK_MC,
        hBLOCK_NC,
        hBLOCK_KC;
    #define BLOCK_NR 12
    #define BLOCK_MR 8
    int BLOCK_MC,
        BLOCK_NC,
        BLOCK_KC;
#else
    //double precission BLIS block sizes fro ARM A-57
    #define dBLOCK_NC 3072
    #define dBLOCK_KC 240
    #define dBLOCK_MC 120
    #define dBLOCK_NR 8
    #define dBLOCK_MR 6
    #define dMAX_THREAD 4

    //simple precission BLIS block sizes fro ARM A-57
    #define BLOCK_NC 3072
    #define BLOCK_KC 640
    #define BLOCK_MC 120
    #define BLOCK_NR 12
    #define BLOCK_MR 8
    #define MAX_THREAD 4

    //half precission BLIS block sizes for NVIDIA Carmel
    #define hBLOCK_NC 3072
    #define hBLOCK_KC 512//684
    #define hBLOCK_MC 576
    #define hBLOCK_NR 8
    #define hBLOCK_MR 24
    #define hMAX_THREAD 4
#endif

       
/**threading functions ****/
struct threadStruct{
    unsigned threads;
    unsigned JC;
    unsigned PC;
    unsigned IC;
    unsigned JR;
    unsigned IR;
    unsigned PR;
};
void getThreadRange(unsigned rangeEnd, unsigned bSize,unsigned nThreads,unsigned *thrStart, unsigned *thrEnd);

        
        
/********double precision gemm********/
void dgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		double  alphap,
		double * A, unsigned int lda,
		double * B, unsigned int ldb,
		double  betap,
		double * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v );

//double precision microkernels
void dgemm_armv8a_asm_6x8(int k, double*  restrict alpha, double* restrict a, double* restrict b, double*    restrict beta, double* restrict c, int rs_c, int cs_c);
void dgemm_ref(int k, int mr_alg, int nr_alg, 	double* restrict alpha, double* restrict a, double* restrict b, double* restrict beta, double* restrict c, int rs_c, int cs_c);

//double precision packing routines
void dPack_A(double *A, unsigned int lda, double *A_pack, unsigned int m, unsigned int k);
void dPack_B(double *B, unsigned int ldb, double *B_pack, unsigned int k, unsigned int n);

/********Simple precision gemm********/
void sgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		float  alphap,
		float * A, unsigned int lda,
		float * B, unsigned int ldb,
		float  betap,
		float * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v);

//Microkernels
void sgemm_armv8a_asm_8x12(int k,float* restrict alpha, float* restrict a, float* restrict b, float* restrict beta, float* restrict c, int rs_c, int cs_c);
void sgemm_armv8a_asm_8x12_v2(int k,float* restrict alpha, float* restrict a, float* restrict b, float* restrict beta, float* restrict c, int rs_c, int cs_c);
void sgemm_armv8a_neon_8x12(int k,float* restrict alpha, float* restrict a, float* restrict b, float* restrict beta, float* restrict c, int rs_c, int cs_c);
void sgemm_ref(int k, int mr_alg, int nr_alg, 	float* restrict alpha, float* restrict a, float* restrict b, float* restrict beta, float* restrict c, int rs_c, int cs_c);

//Packing routines
void sPack_A(float *A, unsigned int lda, float *A_pack, unsigned int m, unsigned int k);
void sPack_B(float *B, unsigned int ldb, float *B_pack, unsigned int k, unsigned int n);


/********Half precision gemm********/
void hgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		_Float16  alphap,
		_Float16 * A, unsigned int lda,
		_Float16 * B, unsigned int ldb,
		_Float16  betap,
		_Float16 * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v,
        struct threadStruct thrSt);

void hsgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		_Float16 alpha,
		_Float16 * A, unsigned int lda,
		_Float16 * B, unsigned int ldb,
		_Float16 beta,
		_Float16 * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v, float *spC_work );

//Microkernels
void hgemm_armv8a_asm_8x24(int k,_Float16* restrict alpha, _Float16* restrict a, _Float16* restrict b, _Float16* restrict beta, _Float16* restrict c, int rs_c, int cs_c);
void hgemm_armv8a_asm_24x8(int k,_Float16* restrict alpha, _Float16* restrict a, _Float16* restrict b, _Float16* restrict beta, _Float16* restrict c, int rs_c, int cs_c);
void hsgemm_armv8a_asm_8x12(int k, _Float16* restrict alpha, _Float16* restrict a,  _Float16* restrict b, _Float16* restrict beta, float* restrict c, int rs_c0, int cs_c0);
void hgemm_ref(int k, int mr_alg, int nr_alg, 	_Float16* restrict alpha, _Float16* restrict a, _Float16* restrict b, _Float16* restrict beta, _Float16* restrict c, int rs_c, int cs_c);
void hsgemm_ref( int k, int mr_alg, int nr_alg, _Float16* restrict alpha, _Float16* restrict a, _Float16* restrict b, _Float16* restrict beta, float* restrict c, int rs_c, int cs_c);


//Packing routines
void hPack_A(_Float16 *A, unsigned int lda, _Float16 *A_pack, unsigned int m, unsigned int k);
void hPack_B(_Float16 *B, unsigned int ldb, _Float16 *B_pack, unsigned int k, unsigned int n);

//Precision change routines
void increasePrecissionV_HS(int  n,  _Float16* restrict buffH, float* restrict buffS);
void decreasePrecissionV_SH( int n, float* restrict buffS,  _Float16* restrict buffH);

/********simple precision convolution gemm********/
void sgemm_conv(unsigned int kh, unsigned int kw, unsigned int c, unsigned int kn,
		float alpha, float * A, 
        unsigned int h, unsigned int w, unsigned int b, unsigned int stride,
		float * B, float beta,
		float * C,
        float * Ac_pack, float * Bc_pack );

//Packing routine
void sPack_im2Col(unsigned int i, unsigned int j,float * restrict B, float * restrict B_pack, unsigned int k, unsigned int n,             
                 unsigned int b, unsigned int c, unsigned int h, unsigned int w, 
                 unsigned int kh, unsigned int kw, unsigned int stride);
            

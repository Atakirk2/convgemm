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
            

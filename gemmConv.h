#include <stdio.h>
#include <stdint.h>




#define dBLOCK_NC 3072
#define dBLOCK_KC 240
#define dBLOCK_MC 120
#define dBLOCK_NR 8
#define dBLOCK_MR 6
#define dMAX_THREAD 4

#define BLOCK_NC 3072
#define BLOCK_KC 640
#define BLOCK_MC 120
#define BLOCK_NR 12
#define BLOCK_MR 8
#define MAX_THREAD 4


void dgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		double  alphap,
		double * A, unsigned int lda,
		double * B, unsigned int ldb,
		double  betap,
		double * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v );

void dgemm_armv8a_asm_6x8(int k, double*  restrict alpha, double* restrict a, double* restrict b, double*    restrict beta, double* restrict c, int rs_c, int cs_c);
void dgemm_ref(int k, int mr_alg, int nr_alg, 	double* restrict alpha, double* restrict a, double* restrict b, double* restrict beta, double* restrict c, int rs_c, int cs_c);

//Simple precission
void sgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		float  alphap,
		float * A, unsigned int lda,
		float * B, unsigned int ldb,
		float  betap,
		float * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v);

void sgemm_armv8a_asm_8x12(int k,float* restrict alpha, float* restrict a, float* restrict b, float* restrict beta, float* restrict c, int rs_c, int cs_c);
void sgemm_ref(int k, int mr_alg, int nr_alg, 	float* restrict alpha, float* restrict a, float* restrict b, float* restrict beta, float* restrict c, int rs_c, int cs_c);


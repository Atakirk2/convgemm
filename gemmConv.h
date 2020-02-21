


//kernel 6x8
#define BLOCK_NC 3072
#define BLOCK_KC 240
#define BLOCK_MC 120
#define BLOCK_NR 8
#define BLOCK_MR 6
#define MAX_THREAD 4



void dgemm_conv(unsigned int m, unsigned int n, unsigned int k,
		double  alphap,
		double * A, unsigned int lda,
		double * B, unsigned int ldb,
		double  betap,
		double * C, unsigned int ldc);

void dgemm_armv8a_asm_6x8(int k, double*  restrict alpha, double* restrict a, double* restrict b, double*    restrict beta, double* restrict c, int rs_c, int cs_c);
void dgemm_ref(int k, int mr_alg, int nr_alg, 	double* restrict alpha, double* restrict a, double* restrict b, double* restrict beta, double* restrict c, int rs_c, int cs_c);

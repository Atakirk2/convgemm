#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#include "gemmConv.h"
//#include "blis.h"

/*
#define BLOCK_NC 4032 //4080
#define BLOCK_KC 256
#define BLOCK_MC 72
#define BLOCK_NR 6
#define BLOCK_MR 8

*/


//OLD definition, mantained for documentation purposes
//void dPack_A(double *A, unsigned int lda, double *A_pack, unsigned int m, unsigned int k, int * n_threads)
void dPack_A(double *A, unsigned int lda, double *A_pack, unsigned int m, unsigned int k)
{
	double *A_pack_local;

	//	printf("soy %d lda %u m %u k %u n_th %d\n", omp_get_thread_num(), lda, m, k, *n_threads);
	//	#pragma omp parallel
	//	#pragma omp single
//		#pragma omp taskloop private(A_pack_local) num_tasks((*n_threads))
//		#pragma omp  parallel for num_threads(*n_threads) private(A_pack_local)
	for(unsigned int ic=0;ic<m;ic+=BLOCK_MR){

		A_pack_local=&A_pack[ic*k];
		unsigned int m_alg=fmin(BLOCK_MR,m-ic);
		for(unsigned int pc=0;pc<k;pc++){

			for(unsigned int ir=0;ir<m_alg;ir++){
				A_pack_local[0]=A[(ic+ir)+pc*lda];
				A_pack_local++;
			}
		}

	}
}


//OLD definition, mantained for documentation purposes
//void dPack_B(double *B, unsigned int ldb, double *B_pack, unsigned int k, unsigned int n, int * n_threads)
void dPack_B(double *B, unsigned int ldb, double *B_pack, unsigned int k, unsigned int n)

{
	double *B_pack_local;

	//	#pragma omp parallel
	//	#pragma omp single
//		#pragma omp taskloop private(B_pack_local) num_tasks((*n_threads))
		////#pragma omp parallel for num_threads(*n_threads) private(B_pack_local)
	for(unsigned int jc=0;jc<n;jc+=BLOCK_NR){

		B_pack_local=&B_pack[jc*k];
		unsigned int n_alg=fmin(BLOCK_NR,n-jc);
		for(unsigned int pc=0;pc<k;pc++){

			for(unsigned int jr=0;jr<n_alg;jr++){
				B_pack_local[0]=B[pc+jc*ldb+jr*ldb];
				B_pack_local++;
			}
		}

	}
}

//OLD definition, mantained for documentation purposes
/*void dgemm_conv(char transA, char transB,
		unsigned int m, unsigned int n, unsigned int k,
		double * &alpha,
		double * A, unsigned int lda,
		double * B, unsigned int ldb,
		double * betap,
		double * C, unsigned int ldc,
		void * Ac_pack_v, void * Bc_pack_v,
		int * jc_threads_nt, int * ic_threads_nt,
		int * jr_threads_nt, int * ir_threads_nt) {*/


void dgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		double alpha,
		double * A, unsigned int lda,
		double * B, unsigned int ldb,
		double beta,
		double * C, unsigned int ldc){
            
	double *Ac, *Bc;
	double *Cc;
	double *Ar, *Br;
	double *Cr;
	double betaInner;


	double *Ac_pack,
           *Bc_pack;

    posix_memalign(&Ac_pack, 4096, BLOCK_MC*BLOCK_KC*sizeof(double));
    posix_memalign(&Bc_pack, 4096, BLOCK_MC*BLOCK_KC*sizeof(double));
    
	//int *a_pack_threads = 1;
	//int *b_pack_threads = 1;


	//printf("alda %u blda %u clda %u\n", lda, ldb, ldc);
	for (unsigned int jc=0; jc<n; jc+=BLOCK_NC) {

		unsigned int n_alg=fmin(BLOCK_NC,n-jc);
		for (unsigned int pc=0; pc<k; pc+=BLOCK_KC) {

			unsigned int k_alg=fmin(BLOCK_KC,k-pc);
			if (pc >= BLOCK_KC) //Check beta
				betaInner=1.0;
			else
				betaInner=beta;

			//total_threads_b = (*ic_threads_nt) * (*jr_threads_nt) * (*ir_threads_nt);
			//b_pack_threads = &total_threads_b;

			Bc=&B[pc+jc*ldb];
			dPack_B(Bc, ldb, Bc_pack, k_alg, n_alg);  //PACK B

			//#pragma omp parallel for num_threads(*ic_threads_nt) private(Ac, Cc, Ar, Br, Cr)
			for (unsigned int ic=0; ic<m; ic+=BLOCK_MC) {

				unsigned int m_alg=fmin(BLOCK_MC,m-ic);
				//				double *Ac_pack_local=&Ac_pack[omp_get_thread_num()*BLOCK_MC*BLOCK_KC]; // Ac pack pointer per Loop 3 thread
				double *Ac_pack_local=Ac_pack; // Ac pack pointer per Loop 3 thread -- antes este en uso

				//total_threads_a = (*jr_threads_nt) * (*ir_threads_nt);
				//a_pack_threads = &total_threads_a;

				Ac=&A[ic+pc*lda];
				dPack_A(Ac,lda,(double*)Ac_pack_local,m_alg,k_alg); //PACK A

				Cc=&C[ic+jc*ldc];

				//printf("jr %d\n", *jr_threads_nt);
				// #pragma omp parallel num_threads(*jr_threads_nt)
				//#pragma omp taskloop private(Ar, Br, Cr) num_tasks((*jr_threads_nt))
			////		#pragma omp  parallel for num_threads(*jr_threads_nt) private(Ar, Br, Cr)
				for(unsigned jr=0;jr<n_alg;jr+=BLOCK_NR){
					//printf("soy %d/%d\n", omp_get_thread_num(), *jr_threads_nt);
					unsigned int nr_alg=fmin(BLOCK_NR,n_alg-jr);
					//printf("secs nr_alg %d n_threads %d\n", nr_alg, *jr_threads_nt);
					for(unsigned int ir=0;ir<m_alg;ir+=BLOCK_MR){
						unsigned int mr_alg=fmin(BLOCK_MR,m_alg-ir);
						Ar=&Ac_pack_local[ir*k_alg];
						Br=&Bc_pack[jr*k_alg];
						Cr=&Cc[ir+jr*ldc];

						if(mr_alg==BLOCK_MR && nr_alg==BLOCK_NR)
						{
							//printf("secs k_alg %d *aphap %p Ar %p Br %p beta %f Cr %p ldc %u\n", k_alg, &alpha, Ar, Br, betaInner, Cr, ldc);	
                            dgemm_armv8a_asm_6x8(k_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
						else{//Micro-kernel cannot be applied
							//printf("secs k_alg %d mr_alg %u nr_alg %u *aphap %p Ar %p Br %p beta %f Cr %p ldc %u\n", k_alg, mr_alg, nr_alg, &alpha, Ar, Br, betaInner, Cr, ldc);	
							dgemm_ref(k_alg,mr_alg,nr_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
					}
				}

			}
		}
	}
}

void sPack_A(float *A, unsigned int lda, float *A_pack, unsigned int m, unsigned int k)
{
	float *A_pack_local;

	//	printf("soy %d lda %u m %u k %u n_th %d\n", omp_get_thread_num(), lda, m, k, *n_threads);
	//	#pragma omp parallel
	//	#pragma omp single
//		#pragma omp taskloop private(A_pack_local) num_tasks((*n_threads))
//		#pragma omp  parallel for num_threads(*n_threads) private(A_pack_local)
	for(unsigned int ic=0;ic<m;ic+=BLOCK_MR){

		A_pack_local=&A_pack[ic*k];
		unsigned int m_alg=fmin(BLOCK_MR,m-ic);
		for(unsigned int pc=0;pc<k;pc++){

			for(unsigned int ir=0;ir<m_alg;ir++){
				A_pack_local[0]=A[(ic+ir)+pc*lda];
				A_pack_local++;
			}
		}

	}
}


void sPack_B(float *B, unsigned int ldb, float *B_pack, unsigned int k, unsigned int n)

{
	float *B_pack_local;

	//	#pragma omp parallel
	//	#pragma omp single
//		#pragma omp taskloop private(B_pack_local) num_tasks((*n_threads))
		////#pragma omp parallel for num_threads(*n_threads) private(B_pack_local)
	for(unsigned int jc=0;jc<n;jc+=BLOCK_NR){

		B_pack_local=&B_pack[jc*k];
		unsigned int n_alg=fmin(BLOCK_NR,n-jc);
		for(unsigned int pc=0;pc<k;pc++){

			for(unsigned int jr=0;jr<n_alg;jr++){
				B_pack_local[0]=B[pc+jc*ldb+jr*ldb];
				B_pack_local++;
			}
		}

	}
}

void sgemm_cust(unsigned int m, unsigned int n, unsigned int k,
		float alpha,
		float * A, unsigned int lda,
		float * B, unsigned int ldb,
		float beta,
		float * C, unsigned int ldc){
            
	float *Ac, *Bc;
	float *Cc;
	float *Ar, *Br;
	float *Cr;
	float betaInner;


	float *Ac_pack,
           *Bc_pack;

    posix_memalign(&Ac_pack, 4096, BLOCK_MC*BLOCK_KC*sizeof(float));
    posix_memalign(&Bc_pack, 4096, BLOCK_MC*BLOCK_KC*sizeof(float));
    
	//int *a_pack_threads = 1;
	//int *b_pack_threads = 1;


	//printf("alda %u blda %u clda %u\n", lda, ldb, ldc);
	for (unsigned int jc=0; jc<n; jc+=BLOCK_NC) {

		unsigned int n_alg=fmin(BLOCK_NC,n-jc);
		for (unsigned int pc=0; pc<k; pc+=BLOCK_KC) {

			unsigned int k_alg=fmin(BLOCK_KC,k-pc);
			if (pc >= BLOCK_KC) //Check beta
				betaInner=1.0;
			else
				betaInner=beta;

			//total_threads_b = (*ic_threads_nt) * (*jr_threads_nt) * (*ir_threads_nt);
			//b_pack_threads = &total_threads_b;

			Bc=&B[pc+jc*ldb];
			sPack_B(Bc, ldb, Bc_pack, k_alg, n_alg);  //PACK B

			//#pragma omp parallel for num_threads(*ic_threads_nt) private(Ac, Cc, Ar, Br, Cr)
			for (unsigned int ic=0; ic<m; ic+=BLOCK_MC) {

				unsigned int m_alg=fmin(BLOCK_MC,m-ic);
				//				float *Ac_pack_local=&Ac_pack[omp_get_thread_num()*BLOCK_MC*BLOCK_KC]; // Ac pack pointer per Loop 3 thread
				float *Ac_pack_local=Ac_pack; // Ac pack pointer per Loop 3 thread -- antes este en uso

				//total_threads_a = (*jr_threads_nt) * (*ir_threads_nt);
				//a_pack_threads = &total_threads_a;

				Ac=&A[ic+pc*lda];
				sPack_A(Ac,lda,(float*)Ac_pack_local,m_alg,k_alg); //PACK A

				Cc=&C[ic+jc*ldc];

				//printf("jr %d\n", *jr_threads_nt);
				// #pragma omp parallel num_threads(*jr_threads_nt)
				//#pragma omp taskloop private(Ar, Br, Cr) num_tasks((*jr_threads_nt))
			////		#pragma omp  parallel for num_threads(*jr_threads_nt) private(Ar, Br, Cr)
				for(unsigned jr=0;jr<n_alg;jr+=BLOCK_NR){
					//printf("soy %d/%d\n", omp_get_thread_num(), *jr_threads_nt);
					unsigned int nr_alg=fmin(BLOCK_NR,n_alg-jr);
					//printf("secs nr_alg %d n_threads %d\n", nr_alg, *jr_threads_nt);
					for(unsigned int ir=0;ir<m_alg;ir+=BLOCK_MR){
						unsigned int mr_alg=fmin(BLOCK_MR,m_alg-ir);
						Ar=&Ac_pack_local[ir*k_alg];
						Br=&Bc_pack[jr*k_alg];
						Cr=&Cc[ir+jr*ldc];

						if(mr_alg==BLOCK_MR && nr_alg==BLOCK_NR)
						{
							//printf("secs k_alg %d *aphap %p Ar %p Br %p beta %f Cr %p ldc %u\n", k_alg, &alpha, Ar, Br, betaInner, Cr, ldc);	
                            sgemm_armv8a_asm_8x12(k_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
						else{//Micro-kernel cannot be applied
							//printf("secs k_alg %d mr_alg %u nr_alg %u *aphap %p Ar %p Br %p beta %f Cr %p ldc %u\n", k_alg, mr_alg, nr_alg, &alpha, Ar, Br, betaInner, Cr, ldc);	
							sgemm_ref(k_alg,mr_alg,nr_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
					}
				}

			}
		}
	}
}


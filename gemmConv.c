#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#include "gemmConv.h"
//#include "blis.h"




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
	for(unsigned int ic=0;ic<m;ic+=dBLOCK_MR){

		A_pack_local=&A_pack[ic*k];
		unsigned int m_alg=fmin(dBLOCK_MR,m-ic);
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
	for(unsigned int jc=0;jc<n;jc+=dBLOCK_NR){

		B_pack_local=&B_pack[jc*k];
		unsigned int n_alg=fmin(dBLOCK_NR,n-jc);
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
		double * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v){
            
	double *Ac, *Bc;
	double *Cc;
	double *Ar, *Br;
	double *Cr;
	double betaInner;


  	double *Ac_pack=(double *)Ac_pack_v;
	double *Bc_pack=(double *)Bc_pack_v;

    
	//int *a_pack_threads = 1;
	//int *b_pack_threads = 1;


	//printf("alda %u blda %u clda %u\n", lda, ldb, ldc);
	for (unsigned int jc=0; jc<n; jc+=dBLOCK_NC) {

		unsigned int n_alg=fmin(dBLOCK_NC,n-jc);
		for (unsigned int pc=0; pc<k; pc+=dBLOCK_KC) {

			unsigned int k_alg=fmin(dBLOCK_KC,k-pc);
			if (pc >= dBLOCK_KC) //Check beta
				betaInner=1.0;
			else
				betaInner=beta;

			//total_threads_b = (*ic_threads_nt) * (*jr_threads_nt) * (*ir_threads_nt);
			//b_pack_threads = &total_threads_b;

			Bc=&B[pc+jc*ldb];
			dPack_B(Bc, ldb, Bc_pack, k_alg, n_alg);  //PACK B

			//#pragma omp parallel for num_threads(*ic_threads_nt) private(Ac, Cc, Ar, Br, Cr)
			for (unsigned int ic=0; ic<m; ic+=dBLOCK_MC) {

				unsigned int m_alg=fmin(dBLOCK_MC,m-ic);
				//				double *Ac_pack_local=&Ac_pack[omp_get_thread_num()*dBLOCK_MC*dBLOCK_KC]; // Ac pack pointer per Loop 3 thread
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
				for(unsigned jr=0;jr<n_alg;jr+=dBLOCK_NR){
					//printf("soy %d/%d\n", omp_get_thread_num(), *jr_threads_nt);
					unsigned int nr_alg=fmin(dBLOCK_NR,n_alg-jr);
					//printf("secs nr_alg %d n_threads %d\n", nr_alg, *jr_threads_nt);
					for(unsigned int ir=0;ir<m_alg;ir+=dBLOCK_MR){
						unsigned int mr_alg=fmin(dBLOCK_MR,m_alg-ir);
						Ar=&Ac_pack_local[ir*k_alg];
						Br=&Bc_pack[jr*k_alg];
						Cr=&Cc[ir+jr*ldc];

						if(mr_alg==dBLOCK_MR && nr_alg==dBLOCK_NR)
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
		#pragma omp  parallel for private(A_pack_local)
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
		#pragma omp parallel for private(B_pack_local)
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
		float * C, unsigned int ldc,
        void * Ac_pack_v, void * Bc_pack_v ){
            
	float *Ac, *Bc;
	float *Cc;
	float *Ar, *Br;
	float *Cr;
	float betaInner;

    
    float *Ac_pack=(float *)Ac_pack_v;
	float *Bc_pack=(float *)Bc_pack_v;
           
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
				#pragma omp  parallel for private(Ar, Br, Cr)
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

void sPack_im2Col(unsigned int i, unsigned int j,float * restrict B, float * restrict B_pack, unsigned int k, unsigned int n,             
                 unsigned int b, unsigned int c, unsigned int h, unsigned int w, 
                 unsigned int kh, unsigned int kw, unsigned int stride)

{
    unsigned int ic,ikw,ikh, //Row related indexes (regarding the phantom matrix)
                 j_local, ib,iw,ih, //Col related indexes (regarding the phantom matrix)
                 pos, pos_ic, pos_ib, pos_ic_ikw; //position on the original image
    unsigned int pos_ic_ini,ikw_ini,ikh_ini,pos_ib_ini,iw_ini,ih_ini; //Initial values of indexes
    
    unsigned int cSize = h*w, //chanel memory leap
                 kSize = kh*kw, //kernel memory leap (single chanel)
                 bSize = c*h*w; //batch memory leap
    
    unsigned int jc,pc,jr; //loop control indexes
	float * restrict B_pack_local;
    
    ic = i/kSize;
    ikw_ini = (i%kSize)/kh;
    ikh_ini = (i%kSize)%kh;
    pos_ic_ini = ic * cSize;



    #pragma omp parallel for private(B_pack_local, j_local,pc,jr,ib,ih_ini, iw_ini, pos_ib_ini,pos_ic,ikw,pos_ic_ikw,ikh,pos_ib,iw,ih,pos) firstprivate(j)
	for(jc=0;jc<n;jc+=BLOCK_NR){

		B_pack_local=&B_pack[jc*k];
		unsigned int n_alg=fmin(BLOCK_NR,n-jc);
        
        j_local = j +jc;
        ib = j_local/cSize;
        iw_ini = (j_local%(cSize))/h;
        ih_ini = (j_local%(cSize))%h;
        pos_ib_ini = ib * bSize;

        

        //ih_ini = ih_ini + jc
        
        pos_ic=pos_ic_ini;
        ikw=ikw_ini;
        pos_ic_ikw = ikw * h + pos_ic;
		for(pc=0,ikh=ikh_ini;pc<k;pc++,ikh++){
            if(ikh==kh)
            {
                ikh=0;
                ikw++;
                pos_ic_ikw += h; //OPT pos_ic_ikw = ikw* h +pos_ic
                if(ikw==kw)
                {
                    ikw=0;
                    pos_ic += cSize;//OPT ic++;pos_ic = ic * cSize;
                    pos_ic_ikw = pos_ic;//OPT pos_ic_ikw = ikw *h + pos_ic;
                }
            }
            
            pos_ib=pos_ib_ini;
            iw=iw_ini;
			for(jr=0,ih=ih_ini;jr<n_alg;jr++,ih++){
                if(ih==h)
                {
                    ih=0;
                    iw++;
                    if(iw==w)
                    {
                        iw=0;
                        pos_ib += bSize;//OPT ib++;pos_in = ib*bSize;
                    }
                }
                //OPT pos = ib * bSize  + ic * cSize + (iw*stride + ikw) *h + (ih * stride+ikh);
                //OPT pos = pos_ib + pos_ic + (iw*stride*h + pos_ikw) + (ih * stride+ikh);
				pos = pos_ib + pos_ic_ikw + iw*stride*h + (ih * stride+ikh);
                
                B_pack_local[0]=B[pos];
				B_pack_local++;
			}
		}
        //ih_ini = ih;
        //iw_ini = iw;
        //pos_ib_ini = pos_ib;
	}
}



void sgemm_conv(unsigned int kh, unsigned int kw, unsigned int c, unsigned int kn,
		float alpha, float * A, 
        unsigned int h, unsigned int w, unsigned int b, unsigned int stride,
		float * B, float beta,
		float * C,
        float * Ac_pack, float * Bc_pack ){
            
	float *Ac, *Bc;
	float *Cc;
	float *Ar, *Br;
	float *Cr;
	float betaInner;

    unsigned int m = kn,
                 n = h*w*b,
                 k = kh*kw*c;
          
    unsigned int lda= kn,
                 ldc= kn;
    
    int num_threads = omp_get_max_threads();
   


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


			sPack_im2Col(pc,jc, B, Bc_pack, k_alg, n_alg, b,c,h,w,kh,kw, stride);  //PACK B

			//#pragma omp parallel for if(m >= (num_threads * BLOCK_MC))  private(Ac, Cc, Ar, Br, Cr)
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
                #pragma omp parallel for  private(Ar, Br, Cr) //if(m < (num_threads * BLOCK_MC))
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

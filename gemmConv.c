/**GEMM and GEMM_conv code
 * 
 * This file contains the implementation of all functions related with the GEMM computation.
 * The code present in this file was derived from a code originally developed for the 
 * following paper: https://doi.org/10.1007/s10586-019-02927-z
 * 
 * @author P. San Juan
 * @date 04/2020
 */

#include "gemmConv.h"





/** Packing of double precision matrix A.
 * 
 * Packs a block of matrix A into a buffer A_pack in the proper data arrengment 
 *  that the microkernel needs.
 * 
 * @param[in] A Pointer pointing to the position of A to start packing.
 * @param[in] lda Leading dimension of matrix a.
 * @param[out] A_pack Buffer containing the portion of A packed.
 * @param[in] m Height of the block to pack.
 * @param[in] k Width of the block to pack.
 */
void dPack_A(double *A, unsigned int lda, double *A_pack, unsigned int m, unsigned int k)
{
	double *A_pack_local;

    #pragma omp  parallel for private(A_pack_local)
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


/** Packing of double precision matrix B.
 * 
 * Packs a block of matrix B into a buffer B_pack in the proper data arrengment 
 *  that the microkernel needs.
 * 
 * @param[in] B Pointer pointing to the position of B to start packing.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[out] B_pack Buffer containing the portion of B packed.
 * @param[in] n Width of the block to pack.
 * @param[in] k Height of the block to pack.
 */
void dPack_B(double *B, unsigned int ldb, double *B_pack, unsigned int k, unsigned int n)

{
	double *B_pack_local;

    #pragma omp parallel for  private(B_pack_local)
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



/** Double precision matrix matrix multiplication.
 * 
 * Performs a matrix matrix product in the form C = alpha * AB + beta * C. Expects atrices stored in column major order.
 * 
 * @param[in] m Number of rows of matrix C and A.
 * @param[in] n Number of columns of matrix C and B.
 * @param[in] k Number of columns of matrix A and rows of matrix B.
 * @param[in] alpha Scalar alpha.
 * @param[in] A Matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] B Matrix B.
 * @param[in] ldB Leading dimension of matrix B.
 * @param[in] beta Scalar beta. 
 * @param[in,out] C Matrix C.
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] Ac_pack_v Workspace for the packing of A (Only ofr allocation purposes).
 * @param[in] Bc_pack_v Workspace for the packing of B (Only ofr allocation purposes).
 */
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



	for (unsigned int jc=0; jc<n; jc+=dBLOCK_NC) {

		unsigned int n_alg=fmin(dBLOCK_NC,n-jc);
		for (unsigned int pc=0; pc<k; pc+=dBLOCK_KC) {

			unsigned int k_alg=fmin(dBLOCK_KC,k-pc);
			if (pc >= dBLOCK_KC) //Check beta
				betaInner=1.0;
			else
				betaInner=beta;

			Bc=&B[pc+jc*ldb];
			dPack_B(Bc, ldb, Bc_pack, k_alg, n_alg);  //PACK B

			
			for (unsigned int ic=0; ic<m; ic+=dBLOCK_MC) {

				unsigned int m_alg=fmin(dBLOCK_MC,m-ic);
				double *Ac_pack_local=Ac_pack; 

				Ac=&A[ic+pc*lda];
				dPack_A(Ac,lda,(double*)Ac_pack_local,m_alg,k_alg); //PACK A

				Cc=&C[ic+jc*ldc];

			    #pragma omp  parallel for private(Ar, Br, Cr)
				for(unsigned jr=0;jr<n_alg;jr+=dBLOCK_NR){
					unsigned int nr_alg=fmin(dBLOCK_NR,n_alg-jr);
					for(unsigned int ir=0;ir<m_alg;ir+=dBLOCK_MR){
						unsigned int mr_alg=fmin(dBLOCK_MR,m_alg-ir);
						Ar=&Ac_pack_local[ir*k_alg];
						Br=&Bc_pack[jr*k_alg];
						Cr=&Cc[ir+jr*ldc];

						if(mr_alg==dBLOCK_MR && nr_alg==dBLOCK_NR)
						{
                            dgemm_armv8a_asm_6x8(k_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
						else{//Micro-kernel cannot be applied
							dgemm_ref(k_alg,mr_alg,nr_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
					}
				}

			}
		}
	}
}

/** Packing of simple precision matrix A.
 * 
 * Packs a block of matrix A into a buffer A_pack in the proper data arrengment 
 *  that the microkernel needs.
 * 
 * @param[in] A Pointer pointing to the position of A to start packing.
 * @param[in] lda Leading dimension of matrix a.
 * @param[in] A_pack Buffer containing the portion of A packed.
 * @param[in] m Height of the block to pack.
 * @param[in] k Width of the block to pack.
 */
void sPack_A(float *A, unsigned int lda, float *A_pack, unsigned int m, unsigned int k)
{
	float *A_pack_local;

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


/** Packing of simple precision matrix B.
 * 
 * Packs a block of matrix B into a buffer B_pack in the proper data arrengment 
 *  that the microkernel needs.
 * 
 * @param[in] B Pointer pointing to the position of B to start packing.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[out] B_pack Buffer containing the portion of B packed.
 * @param[in] n Width of the block to pack.
 * @param[in] k Height of the block to pack.
 */
void sPack_B(float *B, unsigned int ldb, float *B_pack, unsigned int k, unsigned int n)

{
	float *B_pack_local;


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

/** Simple precision matrix matrix multiplication.
 * 
 * Performs a matrix matrix product in the form C = alpha * AB + beta * C. Expects atrices stored in column major order.
 * 
 * @param[in] m Number of rows of matrix C and A.
 * @param[in] n Number of columns of matrix C and B.
 * @param[in] k Number of columns of matrix A and rows of matrix B.
 * @param[in] alpha Scalar alpha .
 * @param[in] A Matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] B Matrix B.
 * @param[in] ldB Leading dimension of matrix B.
 * @param[in] beta Scalar beta. 
 * @param[in,out] C Matrix C.
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] Ac_pack_v Workspace for the packing of A (Only ofr allocation purposes).
 * @param[in] Bc_pack_v Workspace for the packing of B (Only ofr allocation purposes).
 */
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


	for (unsigned int jc=0; jc<n; jc+=BLOCK_NC) {

		unsigned int n_alg=fmin(BLOCK_NC,n-jc);
		for (unsigned int pc=0; pc<k; pc+=BLOCK_KC) {

			unsigned int k_alg=fmin(BLOCK_KC,k-pc);
			if (pc >= BLOCK_KC) //Check beta
				betaInner=1.0;
			else
				betaInner=beta;

			Bc=&B[pc+jc*ldb];
			sPack_B(Bc, ldb, Bc_pack, k_alg, n_alg);  //PACK B

			
			for (unsigned int ic=0; ic<m; ic+=BLOCK_MC) {

				unsigned int m_alg=fmin(BLOCK_MC,m-ic);
				float *Ac_pack_local=Ac_pack; 

				Ac=&A[ic+pc*lda];
				sPack_A(Ac,lda,(float*)Ac_pack_local,m_alg,k_alg); //PACK A

				Cc=&C[ic+jc*ldc];


				#pragma omp  parallel for private(Ar, Br, Cr)
				for(unsigned jr=0;jr<n_alg;jr+=BLOCK_NR){
					unsigned int nr_alg=fmin(BLOCK_NR,n_alg-jr);
					for(unsigned int ir=0;ir<m_alg;ir+=BLOCK_MR){
						unsigned int mr_alg=fmin(BLOCK_MR,m_alg-ir);
						Ar=&Ac_pack_local[ir*k_alg];
						Br=&Bc_pack[jr*k_alg];
						Cr=&Cc[ir+jr*ldc];

						if(mr_alg==BLOCK_MR && nr_alg==BLOCK_NR)
						{
                            sgemm_armv8a_asm_8x12(k_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
						else{//Micro-kernel cannot be applied
							sgemm_ref(k_alg,mr_alg,nr_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						}
					}
				}

			}
		}
	}
}


/** Packing of B + im2Col transform
 * 
 * Packs matrix B = im2Col(In) into the buffer B_pack in the proper data arrengment 
 *  that the microkernel needs. Matrix B does not exist in memory and the data packed 
 *  into B_pack is read from the corresponding positions of the input tensor (In), 
 *  resulting in an on-the-fly im2col transform. 
 * 
 * @param[in] i Row index in matrix B of the first position of the block to pack. 
 * @param[in] j Column index in matrix B of the first position of the block to pack .
 * @param[in] In Input tensor.
 * @param[out] B_pack Buffer containing the portion of B packed.
 * @param[in] k Height of the block to pack.
 * @param[in] n Width of the block to pack.
 * @param[in] b Batch size.
 * @param[in] c number of chanels of input tensor.
 * @param[in] h input tensor hight.
 * @param[in] w input tensor width.
 * @param[in] kh kernel height.
 * @param[in] kw kernel width.
 * @param[in] stride Stride to apply the krnels to the input tensor.
 */
void sPack_im2Col(unsigned int i, unsigned int j,float * restrict In, float * restrict B_pack, unsigned int k, unsigned int n,             
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
                
                B_pack_local[0]=In[pos];
				B_pack_local++;
			}
		}
        //ih_ini = ih;
        //iw_ini = iw;
        //pos_ib_ini = pos_ib;
	}
}


/** Simple precision matrix matrix multiplication with implicit im2col.
 * 
 * Performs a matrix matrix product in the form C = alpha * AB + beta * C, where B = im2col(In). Expects matrices stored in column major order.
 * 
 * @param[in] kh Kernel height.
 * @param[in] kw Kernel width.
 * @param[in] c Number of chanels of input tensor.
 * @param[in] kn Kernel number.
 * @param[in] alpha Scalar alpha.
 * @param[in] A Matrix A. lda assumed as kn.
 * @param[in] h Input tensor hight.
 * @param[in] w Input tensor width.
 * @param[in] b Batch size.
 * @param[in] stride Stride to apply the krnels to the input tensor.
 * @param[in] In 1D-array containing a flattened version of the input tensor.
 * @param[in] beta Scalar beta. 
 * @param[in,out] C Matrix C. ldc asumed as kn.
 * @param[in] Ac_pack Workspace for the packing of A (Only ofr allocation purposes).
 * @param[in] Bc_pack Workspace for the packing of B (Only ofr allocation purposes).
 */
void sgemm_conv(unsigned int kh, unsigned int kw, unsigned int c, unsigned int kn,
		float alpha, float * A, 
        unsigned int h, unsigned int w, unsigned int b, unsigned int stride,
		float * In, float beta,
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
    
	for (unsigned int jc=0; jc<n; jc+=BLOCK_NC) {

		unsigned int n_alg=fmin(BLOCK_NC,n-jc);
		for (unsigned int pc=0; pc<k; pc+=BLOCK_KC) {

			unsigned int k_alg=fmin(BLOCK_KC,k-pc);
			if (pc >= BLOCK_KC) //Check beta
				betaInner=1.0;
			else
				betaInner=beta;

			sPack_im2Col(pc,jc, In, Bc_pack, k_alg, n_alg, b,c,h,w,kh,kw, stride);  //PACK B

			for (unsigned int ic=0; ic<m; ic+=BLOCK_MC) {

				unsigned int m_alg=fmin(BLOCK_MC,m-ic);
				float *Ac_pack_local=Ac_pack; 

				Ac=&A[ic+pc*lda];
				sPack_A(Ac,lda,(float*)Ac_pack_local,m_alg,k_alg); //PACK A

				Cc=&C[ic+jc*ldc];


                #pragma omp parallel for  private(Ar, Br, Cr) 
				for(unsigned jr=0;jr<n_alg;jr+=BLOCK_NR){
					unsigned int nr_alg=fmin(BLOCK_NR,n_alg-jr);
					for(unsigned int ir=0;ir<m_alg;ir+=BLOCK_MR){
						unsigned int mr_alg=fmin(BLOCK_MR,m_alg-ir);
						Ar=&Ac_pack_local[ir*k_alg];
						Br=&Bc_pack[jr*k_alg];
						Cr=&Cc[ir+jr*ldc];

						if(mr_alg==BLOCK_MR && nr_alg==BLOCK_NR)
                            sgemm_armv8a_asm_8x12(k_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
						else//Micro-kernel cannot be applied
							sgemm_ref(k_alg,mr_alg,nr_alg,&alpha,Ar,Br,&betaInner,Cr,1,ldc);
					}
				}

			}
		}
	}
}

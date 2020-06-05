#include <arm_neon.h>
#include <blis.h>

//MR=8 NR=12
void sgemm_armv8a_neon_8x12
     (
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict A,
       float*     restrict B,
       float*     restrict beta,
       float*     restrict C, inc_t rs_c0, inc_t cs_c0//,
      // auxinfo_t* restrict data,
       //cntx_t*    restrict cntx
     )
{

    float32x4_t C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C0A, C0B,
                C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C1A, C1B;
    float32x4_t A0, A1, A2, A3;
    float32x4_t B0, B1, B2;
    float32x4_t vAlpha, vBeta;

    uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0 -4;
    uint64_t l;
    float32_t* aPtr = A, 
              *bPtr = B,
              *clPtr = C,
              *csPtr = C;
    
              
    //char charA='A', charB = 'B', charC='C';
   // printf("K = %lu, cs_c = %lu, alpha = %g, beta = %g\n",k,cs_c, *alpha, *beta);
    //print_matrix(&charA, 8, k, A, 8 );
    //print_matrix(&charB, 12, k, B, 12 );
	//print_matrix(&charC, 8, 12, C, cs_c0 );
    
	C00=vmovq_n_f32(0);
    C10=vmovq_n_f32(0);
    C01=vmovq_n_f32(0);
    C11=vmovq_n_f32(0);
    C02=vmovq_n_f32(0);
    C12=vmovq_n_f32(0);
    C03=vmovq_n_f32(0);
    C13=vmovq_n_f32(0);
    C04=vmovq_n_f32(0);
    C14=vmovq_n_f32(0);
    C05=vmovq_n_f32(0);
    C15=vmovq_n_f32(0);
    C06=vmovq_n_f32(0);
    C16=vmovq_n_f32(0);
    C07=vmovq_n_f32(0);
    C17=vmovq_n_f32(0);
    C08=vmovq_n_f32(0);
    C18=vmovq_n_f32(0);
    C09=vmovq_n_f32(0);
    C19=vmovq_n_f32(0);
    C0A=vmovq_n_f32(0);
    C1A=vmovq_n_f32(0);
    C0B=vmovq_n_f32(0);
    C1B=vmovq_n_f32(0);



	/* Perform a series of k rank-1 updates into ab. */
	while(k_iter > 0)
	{
        //iter k0%1
        A0=vld1q_f32(aPtr); aPtr+=4;
        A1=vld1q_f32(aPtr); aPtr+=4;
        
        B0 = vld1q_f32(bPtr); bPtr+=4;
        C00 = vfmaq_laneq_f32(C00, A0, B0, 0);   
        C10 = vfmaq_laneq_f32(C10, A1, B0, 0);
        C01 = vfmaq_laneq_f32(C01, A0, B0, 1);
        C11 = vfmaq_laneq_f32(C11, A1, B0, 1);
        C02 = vfmaq_laneq_f32(C02, A0, B0, 2);
        C12 = vfmaq_laneq_f32(C12, A1, B0, 2);
        C03 = vfmaq_laneq_f32(C03, A0, B0, 3);
        C13 = vfmaq_laneq_f32(C13, A1, B0, 3);
        
        B1 = vld1q_f32(bPtr); bPtr+=4;
        C04 = vfmaq_laneq_f32(C04, A0, B1, 0);
        C14 = vfmaq_laneq_f32(C14, A1, B1, 0);
        C05 = vfmaq_laneq_f32(C05, A0, B1, 1);
        C15 = vfmaq_laneq_f32(C15, A1, B1, 1);
        C06 = vfmaq_laneq_f32(C06, A0, B1, 2);
        C16 = vfmaq_laneq_f32(C16, A1, B1, 2);
        C07 = vfmaq_laneq_f32(C07, A0, B1, 3);
        C17 = vfmaq_laneq_f32(C17, A1, B1, 3);
        
        B2 = vld1q_f32(bPtr); bPtr+=4;
        C08 = vfmaq_laneq_f32(C08, A0, B2, 0);
        C18 = vfmaq_laneq_f32(C18, A1, B2, 0);
        C09 = vfmaq_laneq_f32(C09, A0, B2, 1);
        C19 = vfmaq_laneq_f32(C19, A1, B2, 1);
        C0A = vfmaq_laneq_f32(C0A, A0, B2, 2);
        C1A = vfmaq_laneq_f32(C1A, A1, B2, 2);
        C0B = vfmaq_laneq_f32(C0B, A0, B2, 3);
        C1B = vfmaq_laneq_f32(C1B, A1, B2, 3);
        
        
        //iter k0%2
        A2=vld1q_f32(aPtr); aPtr+=4;
        A3=vld1q_f32(aPtr); aPtr+=4;
        
        B0 = vld1q_f32(bPtr); bPtr+=4;
        C00 = vfmaq_laneq_f32(C00, A2, B0, 0);   
        C10 = vfmaq_laneq_f32(C10, A3, B0, 0);
        C01 = vfmaq_laneq_f32(C01, A2, B0, 1);
        C11 = vfmaq_laneq_f32(C11, A3, B0, 1);
        C02 = vfmaq_laneq_f32(C02, A2, B0, 2);
        C12 = vfmaq_laneq_f32(C12, A3, B0, 2);
        C03 = vfmaq_laneq_f32(C03, A2, B0, 3);
        C13 = vfmaq_laneq_f32(C13, A3, B0, 3);
        
        B1 = vld1q_f32(bPtr); bPtr+=4;
        C04 = vfmaq_laneq_f32(C04, A2, B1, 0);
        C14 = vfmaq_laneq_f32(C14, A3, B1, 0);
        C05 = vfmaq_laneq_f32(C05, A2, B1, 1);
        C15 = vfmaq_laneq_f32(C15, A3, B1, 1);
        C06 = vfmaq_laneq_f32(C06, A2, B1, 2);
        C16 = vfmaq_laneq_f32(C16, A3, B1, 2);
        C07 = vfmaq_laneq_f32(C07, A2, B1, 3);
        C17 = vfmaq_laneq_f32(C17, A3, B1, 3);
        
        B2 = vld1q_f32(bPtr); bPtr+=4;
        C08 = vfmaq_laneq_f32(C08, A2, B2, 0);
        C18 = vfmaq_laneq_f32(C18, A3, B2, 0);
        C09 = vfmaq_laneq_f32(C09, A2, B2, 1);
        C19 = vfmaq_laneq_f32(C19, A3, B2, 1);
        C0A = vfmaq_laneq_f32(C0A, A2, B2, 2);
        C1A = vfmaq_laneq_f32(C1A, A3, B2, 2);
        C0B = vfmaq_laneq_f32(C0B, A2, B2, 3);
        C1B = vfmaq_laneq_f32(C1B, A3, B2, 3);
        
        
        //iter k0%3
        A0=vld1q_f32(aPtr); aPtr+=4;
        A1=vld1q_f32(aPtr); aPtr+=4;
        
        B0 = vld1q_f32(bPtr); bPtr+=4;
        C00 = vfmaq_laneq_f32(C00, A0, B0, 0);   
        C10 = vfmaq_laneq_f32(C10, A1, B0, 0);
        C01 = vfmaq_laneq_f32(C01, A0, B0, 1);
        C11 = vfmaq_laneq_f32(C11, A1, B0, 1);
        C02 = vfmaq_laneq_f32(C02, A0, B0, 2);
        C12 = vfmaq_laneq_f32(C12, A1, B0, 2);
        C03 = vfmaq_laneq_f32(C03, A0, B0, 3);
        C13 = vfmaq_laneq_f32(C13, A1, B0, 3);
        
        B1 = vld1q_f32(bPtr); bPtr+=4;
        C04 = vfmaq_laneq_f32(C04, A0, B1, 0);
        C14 = vfmaq_laneq_f32(C14, A1, B1, 0);
        C05 = vfmaq_laneq_f32(C05, A0, B1, 1);
        C15 = vfmaq_laneq_f32(C15, A1, B1, 1);
        C06 = vfmaq_laneq_f32(C06, A0, B1, 2);
        C16 = vfmaq_laneq_f32(C16, A1, B1, 2);
        C07 = vfmaq_laneq_f32(C07, A0, B1, 3);
        C17 = vfmaq_laneq_f32(C17, A1, B1, 3);
        
        B2 = vld1q_f32(bPtr); bPtr+=4;
        C08 = vfmaq_laneq_f32(C08, A0, B2, 0);
        C18 = vfmaq_laneq_f32(C18, A1, B2, 0);
        C09 = vfmaq_laneq_f32(C09, A0, B2, 1);
        C19 = vfmaq_laneq_f32(C19, A1, B2, 1);
        C0A = vfmaq_laneq_f32(C0A, A0, B2, 2);
        C1A = vfmaq_laneq_f32(C1A, A1, B2, 2);
        C0B = vfmaq_laneq_f32(C0B, A0, B2, 3);
        C1B = vfmaq_laneq_f32(C1B, A1, B2, 3);
        
        //iter k0%4
        A2=vld1q_f32(aPtr); aPtr+=4;
        A3=vld1q_f32(aPtr); aPtr+=4;
        
        B0 = vld1q_f32(bPtr); bPtr+=4;
        C00 = vfmaq_laneq_f32(C00, A2, B0, 0);   
        C10 = vfmaq_laneq_f32(C10, A3, B0, 0);
        C01 = vfmaq_laneq_f32(C01, A2, B0, 1);
        C11 = vfmaq_laneq_f32(C11, A3, B0, 1);
        C02 = vfmaq_laneq_f32(C02, A2, B0, 2);
        C12 = vfmaq_laneq_f32(C12, A3, B0, 2);
        C03 = vfmaq_laneq_f32(C03, A2, B0, 3);
        C13 = vfmaq_laneq_f32(C13, A3, B0, 3);
        
        B1 = vld1q_f32(bPtr); bPtr+=4;
        C04 = vfmaq_laneq_f32(C04, A2, B1, 0);
        C14 = vfmaq_laneq_f32(C14, A3, B1, 0);
        C05 = vfmaq_laneq_f32(C05, A2, B1, 1);
        C15 = vfmaq_laneq_f32(C15, A3, B1, 1);
        C06 = vfmaq_laneq_f32(C06, A2, B1, 2);
        C16 = vfmaq_laneq_f32(C16, A3, B1, 2);
        C07 = vfmaq_laneq_f32(C07, A2, B1, 3);
        C17 = vfmaq_laneq_f32(C17, A3, B1, 3);
        
        B2 = vld1q_f32(bPtr); bPtr+=4;
        C08 = vfmaq_laneq_f32(C08, A2, B2, 0);
        C18 = vfmaq_laneq_f32(C18, A3, B2, 0);
        C09 = vfmaq_laneq_f32(C09, A2, B2, 1);
        C19 = vfmaq_laneq_f32(C19, A3, B2, 1);
        C0A = vfmaq_laneq_f32(C0A, A2, B2, 2);
        C1A = vfmaq_laneq_f32(C1A, A3, B2, 2);
        C0B = vfmaq_laneq_f32(C0B, A2, B2, 3);
        C1B = vfmaq_laneq_f32(C1B, A3, B2, 3);
        
        k_iter--;
	} 
	
	// Remaining iteration not multiple of urolling(4)
	
	while(k_left > 0)
    {
        A0=vld1q_f32(aPtr); aPtr+=4;
        A1=vld1q_f32(aPtr); aPtr+=4;
        
        B0 = vld1q_f32(bPtr); bPtr+=4;
        C00 = vfmaq_laneq_f32(C00, A0, B0, 0);   
        C10 = vfmaq_laneq_f32(C10, A1, B0, 0);
        C01 = vfmaq_laneq_f32(C01, A0, B0, 1);
        C11 = vfmaq_laneq_f32(C11, A1, B0, 1);
        C02 = vfmaq_laneq_f32(C02, A0, B0, 2);
        C12 = vfmaq_laneq_f32(C12, A1, B0, 2);
        C03 = vfmaq_laneq_f32(C03, A0, B0, 3);
        C13 = vfmaq_laneq_f32(C13, A1, B0, 3);
        
        B1 = vld1q_f32(bPtr); bPtr+=4;
        C04 = vfmaq_laneq_f32(C04, A0, B1, 0);
        C14 = vfmaq_laneq_f32(C14, A1, B1, 0);
        C05 = vfmaq_laneq_f32(C05, A0, B1, 1);
        C15 = vfmaq_laneq_f32(C15, A1, B1, 1);
        C06 = vfmaq_laneq_f32(C06, A0, B1, 2);
        C16 = vfmaq_laneq_f32(C16, A1, B1, 2);
        C07 = vfmaq_laneq_f32(C07, A0, B1, 3);
        C17 = vfmaq_laneq_f32(C17, A1, B1, 3);
        
        B2 = vld1q_f32(bPtr); bPtr+=4;
        C08 = vfmaq_laneq_f32(C08, A0, B2, 0);
        C18 = vfmaq_laneq_f32(C18, A1, B2, 0);
        C09 = vfmaq_laneq_f32(C09, A0, B2, 1);
        C19 = vfmaq_laneq_f32(C19, A1, B2, 1);
        C0A = vfmaq_laneq_f32(C0A, A0, B2, 2);
        C1A = vfmaq_laneq_f32(C1A, A1, B2, 2);
        C0B = vfmaq_laneq_f32(C0B, A0, B2, 3);
        C1B = vfmaq_laneq_f32(C1B, A1, B2, 3);
        
        k_left--;
    }
	
	//Scale and Write results 
     
    vAlpha = vld1q_lane_f32(alpha,vAlpha,0);
    vBeta = vld1q_lane_f32(beta,vBeta,0);
     
    A0 =vmovq_n_f32(0); //Clean vector registers
    A1 =vmovq_n_f32(0);
    A2 =vmovq_n_f32(0);
    A3 =vmovq_n_f32(0);
    B0 =vmovq_n_f32(0);
    B1 =vmovq_n_f32(0);
    
    if(beta!=0) 
    {
        A0 = vld1q_f32(clPtr); clPtr+=4; //Load column 0 of C
        A1 = vld1q_f32(clPtr); clPtr+=cs_c;
	    A2 = vld1q_f32(clPtr); clPtr+= 4; //Load column 1 of C
        A3 = vld1q_f32(clPtr); clPtr+=cs_c;
        B0 = vld1q_f32(clPtr); clPtr+= 4; //Load column 2 of C
        B1 = vld1q_f32(clPtr); clPtr+=cs_c;
      
        A0 = vmulq_laneq_f32(A0, vBeta, 0); //Scale by beta
        A1 = vmulq_laneq_f32(A1, vBeta, 0);
        A2 = vmulq_laneq_f32(A2, vBeta, 0);
        A3 = vmulq_laneq_f32(A3, vBeta, 0);
        B0 = vmulq_laneq_f32(B0, vBeta, 0);
        B1 = vmulq_laneq_f32(B1, vBeta, 0);
    }
	
    A0 = vfmaq_laneq_f32(A0, C00, vAlpha, 0); //Scale by alpha
    A1 = vfmaq_laneq_f32(A1, C10, vAlpha, 0);
    A2 = vfmaq_laneq_f32(A2, C01, vAlpha, 0);
    A3 = vfmaq_laneq_f32(A3, C11, vAlpha, 0);
    B0 = vfmaq_laneq_f32(B0, C02, vAlpha, 0);
    B1 = vfmaq_laneq_f32(B1, C12, vAlpha, 0);
      
    vst1q_f32(csPtr, A0); csPtr += 4; //Store column 0 of C
    vst1q_f32(csPtr, A1); csPtr +=cs_c;
    vst1q_f32(csPtr, A2); csPtr += 4; //Store column 1 of C
    vst1q_f32(csPtr, A3); csPtr +=cs_c;
    vst1q_f32(csPtr, B0); csPtr += 4; //Store column 2 of C
    vst1q_f32(csPtr, B1); csPtr +=cs_c;
    
  
    C00 =vmovq_n_f32(0); //Clean vector registers
    C10 =vmovq_n_f32(0);
    C01 =vmovq_n_f32(0);
    C11 =vmovq_n_f32(0);
    C02 =vmovq_n_f32(0);
    C12 =vmovq_n_f32(0);
    
    
    if(beta!=0) 
    {
        C00 = vld1q_f32(clPtr); clPtr+= 4; //Load column 3 of C
        C10 = vld1q_f32(clPtr); clPtr+=cs_c;
	    C01 = vld1q_f32(clPtr); clPtr+= 4; //Load column 4 of C
        C11 = vld1q_f32(clPtr); clPtr+=cs_c;
        C02 = vld1q_f32(clPtr); clPtr+= 4; //Load column 5 of C
        C12 = vld1q_f32(clPtr); clPtr+=cs_c;
      
        C00 = vmulq_laneq_f32(C00, vBeta, 0); //Scale by beta
        C10 = vmulq_laneq_f32(C10, vBeta, 0);
        C01 = vmulq_laneq_f32(C01, vBeta, 0);
        C11 = vmulq_laneq_f32(C11, vBeta, 0);
        C02 = vmulq_laneq_f32(C02, vBeta, 0);
        C12 = vmulq_laneq_f32(C12, vBeta, 0);
    }
    
    C00 = vfmaq_laneq_f32(C00, C03, vAlpha, 0); //Scale by alpha
    C10 = vfmaq_laneq_f32(C10, C13, vAlpha, 0);
    C01 = vfmaq_laneq_f32(C01, C04, vAlpha, 0);
    C11 = vfmaq_laneq_f32(C11, C14, vAlpha, 0);
    C02 = vfmaq_laneq_f32(C02, C05, vAlpha, 0);
    C12 = vfmaq_laneq_f32(C12, C15, vAlpha, 0);
      
    vst1q_f32(csPtr, C00); csPtr += 4; //Store column 3 of C
    vst1q_f32(csPtr, C10); csPtr +=cs_c;
    vst1q_f32(csPtr, C01); csPtr += 4; //Store column 4 of C
    vst1q_f32(csPtr, C11); csPtr +=cs_c;
    vst1q_f32(csPtr, C02); csPtr += 4; //Store column 5 of C
    vst1q_f32(csPtr, C12); csPtr +=cs_c;
    
    
    
    A0 =vmovq_n_f32(0); //Clean vector registers
    A1 =vmovq_n_f32(0);
    A2 =vmovq_n_f32(0);
    A3 =vmovq_n_f32(0);
    B0 =vmovq_n_f32(0);
    B1 =vmovq_n_f32(0);
    
    
    if(beta!=0) 
    {
        A0 = vld1q_f32(clPtr); clPtr+= 4; //Load column 6 of C
        A1 = vld1q_f32(clPtr); clPtr+=cs_c;
	    A2 = vld1q_f32(clPtr); clPtr+= 4; //Load column 7 of C
        A3 = vld1q_f32(clPtr); clPtr+=cs_c;
        B0 = vld1q_f32(clPtr); clPtr+= 4; //Load column 8 of C
        B1 = vld1q_f32(clPtr); clPtr+=cs_c;
      
        A0 = vmulq_laneq_f32(A0, vBeta, 0); //Scale by beta
        A1 = vmulq_laneq_f32(A1, vBeta, 0);
        A2 = vmulq_laneq_f32(A2, vBeta, 0);
        A3 = vmulq_laneq_f32(A3, vBeta, 0);
        B0 = vmulq_laneq_f32(B0, vBeta, 0);
        B1 = vmulq_laneq_f32(B1, vBeta, 0);
    }
	
    A0 = vfmaq_laneq_f32(A0, C06, vAlpha, 0); //Scale by alpha
    A1 = vfmaq_laneq_f32(A1, C16, vAlpha, 0);
     A2 = vfmaq_laneq_f32(A2, C07, vAlpha, 0);
    A3 = vfmaq_laneq_f32(A3, C17, vAlpha, 0);
    B0 = vfmaq_laneq_f32(B0, C08, vAlpha, 0);
    B1 = vfmaq_laneq_f32(B1, C18, vAlpha, 0);
      
    vst1q_f32(csPtr, A0); csPtr += 4; //Store column 6 of C
    vst1q_f32(csPtr, A1); csPtr +=cs_c;
    vst1q_f32(csPtr, A2); csPtr += 4; //Store column 7 of C
    vst1q_f32(csPtr, A3); csPtr +=cs_c;
    vst1q_f32(csPtr, B0); csPtr += 4; //Store column 8 of C
    vst1q_f32(csPtr, B1); csPtr +=cs_c;
    
  
    C00 =vmovq_n_f32(0); //Clean vector registers
    C10 =vmovq_n_f32(0);
    C01 =vmovq_n_f32(0);
    C11 =vmovq_n_f32(0);
    C02 =vmovq_n_f32(0);
    C12 =vmovq_n_f32(0);
    
    
    if(beta!=0) 
    {
        C00 = vld1q_f32(clPtr); clPtr+= 4; //Load column 9 of C
        C10 = vld1q_f32(clPtr); clPtr+=cs_c;
	    C01 = vld1q_f32(clPtr); clPtr+= 4; //Load column 10 of C
        C11 = vld1q_f32(clPtr); clPtr+=cs_c;
        C02 = vld1q_f32(clPtr); clPtr+= 4; //Load column 11 of C
        C12 = vld1q_f32(clPtr); clPtr+=cs_c;
      
        C00 = vmulq_laneq_f32(C00, vBeta, 0); //Scale by beta
        C10 = vmulq_laneq_f32(C10, vBeta, 0);
        C01 = vmulq_laneq_f32(C01, vBeta, 0);
        C11 = vmulq_laneq_f32(C11, vBeta, 0);
        C02 = vmulq_laneq_f32(C02, vBeta, 0);
        C12 = vmulq_laneq_f32(C12, vBeta, 0);
    }
    
    C00 = vfmaq_laneq_f32(C00, C09, vAlpha, 0); //Scale by alpha
    C10 = vfmaq_laneq_f32(C10, C19, vAlpha, 0);
    C01 = vfmaq_laneq_f32(C01, C0A, vAlpha, 0);
    C11 = vfmaq_laneq_f32(C11, C1A, vAlpha, 0);
    C02 = vfmaq_laneq_f32(C02, C0B, vAlpha, 0);
    C12 = vfmaq_laneq_f32(C12, C1B, vAlpha, 0);
      
    vst1q_f32(csPtr, C00); csPtr += 4; //Store column 9 of C
    vst1q_f32(csPtr, C10); csPtr +=cs_c;
    vst1q_f32(csPtr, C01); csPtr += 4; //Store column 10 of C
    vst1q_f32(csPtr, C11); csPtr +=cs_c;
    vst1q_f32(csPtr, C02); csPtr += 4; //Store column 11 of C
    vst1q_f32(csPtr, C12); csPtr +=cs_c;
    
    
    //print_matrix(&charC, 8, 12, C, cs_c0 );
    
}

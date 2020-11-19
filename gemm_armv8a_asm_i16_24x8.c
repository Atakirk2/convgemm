#include <blis.h>

void i16gemm_armv8a_asm_24x8
     (
       dim_t               k0,
       int16_t*     restrict alpha,
       int16_t*     restrict a,
       int16_t*     restrict b,
       int16_t*     restrict beta,
       int16_t*     restrict c, inc_t rs_c0, inc_t cs_c0//,
      // auxinfo_t* restrict data,
       //cntx_t*    restrict cntx
     )
{
	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

__asm__ volatile 
(
"                                            \n\t"
"                                            \n\t"
" ldr x0,%[aaddr]                            \n\t" // Load address of A. 
" ldr x1,%[baddr]                            \n\t" // Load address of B.
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
//" ldr x3,%[a_next]                           \n\t" // Pointer to next block of A.
//" ldr x4,%[b_next]                           \n\t" // Pointer to next pointer of B.
"                                            \n\t"
" ldr x5,%[k_iter]                           \n\t" // Number of unrolled iterations (k_iter).
" ldr x6,%[k_left]                           \n\t" // Number of remaining iterations (k_left).
"                                            \n\t" 
" ldr x7,%[alpha]                            \n\t" // Alpha address.      
" ldr x8,%[beta]                             \n\t" // Beta address.     
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c.
" lsl x10,x9,#1                              \n\t" // cs_c * sizeof(fp16) -- AUX.
"                                            \n\t" 
//" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#1                             \n\t" // rs_c * sizeof(fp16).
"                                            \n\t"
" add x16,x2,x10                             \n\t" //Load address Column 1 of C
" add x17,x16,x10                            \n\t" //Load address Column 2 of C
" add x18,x17,x10                            \n\t" //Load address Column 3 of C
" add x19,x18,x10                            \n\t" //Load address Column 4 of C
" add x20,x19,x10                            \n\t" //Load address Column 5 of C
" add x21,x20,x10                            \n\t" //Load address Column 6 of C
" add x22,x21,x10                            \n\t" //Load address Column 7 of C


"                                            \n\t"
" ld1 {v0.8h,v1.8h, v2.8h}, [x0]                   \n\t" //Load a
"                                            \n\t"
" ld1 {v6.8h}, [x1]            \n\t" //Load b
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x16]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x17]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x18]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x19]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.


"                                            \n\t"
" dup  v8.8h, wzr                            \n\t" // Vectors for accummulating column 0
" dup  v9.8h, wzr                            \n\t" 
" dup  v10.8h, wzr                           \n\t"
" dup  v11.8h, wzr                           \n\t" // Vector for accummulating column 1
" dup  v12.8h, wzr                           \n\t" 
" dup  v13.8h, wzr                           \n\t" 
"                                            \n\t"
" dup  v14.8h, wzr                           \n\t" // Vector for accummulating column 2
" prfm    PLDL1KEEP, [x0, #64]              \n\t"
" dup  v15.8h, wzr                           \n\t" 
" prfm    PLDL1KEEP, [x0, #128]              \n\t"
" dup  v16.8h, wzr                           \n\t" 
" dup  v17.8h, wzr                           \n\t" // Vector for accummulating column 3
" dup  v18.8h, wzr                           \n\t" 
" dup  v19.8h, wzr                           \n\t" 
"                                            \n\t"
" dup  v20.8h, wzr                           \n\t" // Vector for accummulating column 4 
" dup  v21.8h, wzr                           \n\t" 
" dup  v22.8h, wzr                           \n\t" 
" dup  v23.8h, wzr                           \n\t" // Vector for accummulating column 5
" dup  v24.8h, wzr                           \n\t" 
" dup  v25.8h, wzr                           \n\t" 
"                                            \n\t"
" dup  v26.8h, wzr                           \n\t" // Vector for accummulating column 6 
" dup  v27.8h, wzr                           \n\t" 
" dup  v28.8h, wzr                           \n\t" 
" dup  v29.8h, wzr                           \n\t" // Vector for accummulating column 7
" dup  v30.8h, wzr                           \n\t"  
" dup  v31.8h, wzr                           \n\t" 
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .SCONSIDERKLEFT                        \n\t"
"                                            \n\t"
"add x0, x0, #48                             \n\t" //update address of A
"add x1, x1, #16                             \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .SLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" .SLOOPKITER:                               \n\t" // Body of the k_iter loop.
"                                            \n\t"

" mla v8.8h, v0.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v0.8h,v6.h[1]                  \n\t" // Accummulate.
" ld1 {v3.8h, v4.8h, v5.8h}, [x0],#48                      \n\t"
" mla v14.8h,v0.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v0.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v0.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v0.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v0.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v0.8h,v6.h[7]                  \n\t" // Accummulate.
" ld1 {v7.8h}, [x1],#16        \n\t"
"                                            \n\t"
" prfm    PLDL1KEEP, [x1, #32]              \n\t" 
" mla v9.8h, v1.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v2.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v2.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v2.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v2.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v2.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v2.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v2.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v2.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" mla v8.8h, v3.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v3.8h,v7.h[1]                  \n\t" // Accummulate.
" ld1 {v0.8h, v1.8h, v2.8h}, [x0],#48                      \n\t"

" mla v14.8h,v3.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v3.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v3.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v3.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v3.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v3.8h,v7.h[7]                  \n\t" // Accummulate.
" ld1 {v6.8h}, [x1],#16        \n\t"
"                                            \n\t"
" prfm    PLDL1KEEP, [x0, #48]              \n\t"
" mla v9.8h, v4.8h,v7.h[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #112]              \n\t"
" mla v12.8h,v4.8h,v7.h[1]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #176]              \n\t"
" mla v15.8h,v4.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v4.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v4.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v4.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v4.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v4.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v5.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v5.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v5.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v5.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v5.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v5.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v5.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v5.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" mla v8.8h, v0.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v0.8h,v6.h[1]                  \n\t" // Accummulate.
" ld1 {v3.8h, v4.8h, v5.8h}, [x0],#48                      \n\t"

" mla v14.8h,v0.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v0.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v0.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v0.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v0.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v0.8h,v6.h[7]                  \n\t" // Accummulate.
" ld1 {v7.8h}, [x1],#16        \n\t"
"                                            \n\t"
" mla v9.8h, v1.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v2.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v2.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v2.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v2.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v2.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v2.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v2.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v2.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" mla v8.8h, v3.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v3.8h,v7.h[1]                  \n\t" // Accummulate.
" ld1 {v0.8h, v1.8h, v2.8h}, [x0],#48                      \n\t"
" mla v14.8h,v3.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v3.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v3.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v3.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v3.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v3.8h,v7.h[7]                  \n\t" // Accummulate.
" ld1 {v6.8h}, [x1],#16        \n\t"
"                                            \n\t"
" mla v9.8h, v4.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v4.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v4.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v4.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v4.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v4.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v4.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v4.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v5.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v5.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v5.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v5.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v5.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v5.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v5.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v5.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 4
" sub x5,x5,1                                \n\t" // i-=1.
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .SLOOPKITER                            \n\t"
"                                            \n\t" 
" .SLASTITER:                                \n\t" // Last iteration of k_iter loop.
"                                            \n\t" 
" mla v8.8h, v0.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v0.8h,v6.h[1]                  \n\t" // Accummulate.
" ld1 {v3.8h, v4.8h, v5.8h}, [x0],#48                      \n\t"

" mla v14.8h,v0.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v0.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v0.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v0.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v0.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v0.8h,v6.h[7]                  \n\t" // Accummulate.
" ld1 {v7.8h}, [x1],#16        \n\t"
"                                            \n\t"
" mla v9.8h, v1.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v2.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v2.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v2.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v2.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v2.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v2.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v2.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v2.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" mla v8.8h, v3.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v3.8h,v7.h[1]                  \n\t" // Accummulate.
" ld1 {v0.8h, v1.8h, v2.8h}, [x0],#48                      \n\t"

" mla v14.8h,v3.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v3.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v3.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v3.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v3.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v3.8h,v7.h[7]                  \n\t" // Accummulate.
" ld1 {v6.8h}, [x1],#16        \n\t"
"                                            \n\t"
" mla v9.8h, v4.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v4.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v4.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v4.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v4.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v4.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v4.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v4.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v5.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v5.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v5.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v5.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v5.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v5.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v5.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v5.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" mla v8.8h, v0.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v0.8h,v6.h[1]                  \n\t" // Accummulate.
" ld1 {v3.8h, v4.8h, v5.8h}, [x0],#48                      \n\t"

" mla v14.8h,v0.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v0.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v0.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v0.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v0.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v0.8h,v6.h[7]                  \n\t" // Accummulate.
" ld1 {v7.8h}, [x1],#16        \n\t"
"                                            \n\t"
" mla v9.8h, v1.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v2.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v2.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v2.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v2.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v2.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v2.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v2.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v2.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" mla v8.8h, v3.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v3.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v14.8h,v3.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v3.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v3.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v3.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v3.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v3.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v9.8h, v4.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v4.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v4.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v4.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v4.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v4.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v4.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v4.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v5.8h,v7.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v5.8h,v7.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v5.8h,v7.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v5.8h,v7.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v5.8h,v7.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v5.8h,v7.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v5.8h,v7.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v5.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 4
"                                            \n\t"
" .SCONSIDERKLEFT:                           \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .SPOSTACCUM                            \n\t" // else, we enter the k_left loop.
"                                            \n\t"
" .SLOOPKLEFT:                               \n\t" // Body of the left iterations
"                                            \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t" // i = i-1.
"                                            \n\t"
" ld1 {v0.8h, v1.8h, v2.8h}, [x0],#48                      \n\t"
" ld1 {v6.8h}, [x1],#16        \n\t"
" mla v8.8h, v0.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v11.8h,v0.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v14.8h,v0.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v17.8h,v0.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v20.8h,v0.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v23.8h,v0.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v26.8h,v0.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v29.8h,v0.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v9.8h, v1.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v12.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v15.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v18.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v21.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v24.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v27.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v30.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" mla v10.8h,v2.8h,v6.h[0]                  \n\t" // Accummulate.
" mla v13.8h,v2.8h,v6.h[1]                  \n\t" // Accummulate.
" mla v16.8h,v2.8h,v6.h[2]                  \n\t" // Accummulate.
" mla v19.8h,v2.8h,v6.h[3]                  \n\t" // Accummulate.
" mla v22.8h,v2.8h,v6.h[4]                  \n\t" // Accummulate.
" mla v25.8h,v2.8h,v6.h[5]                  \n\t" // Accummulate.
" mla v28.8h,v2.8h,v6.h[6]                  \n\t" // Accummulate.
" mla v31.8h,v2.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .SLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" .SPOSTACCUM:                               \n\t"
"                                            \n\t"
//" ld1r {v6.8h},[x7]                          \n\t" // Load alpha.
//" ld1r {v7.8h},[x8]                          \n\t" // Load beta
" ld1 {v7.h}[1],[x7]                          \n\t" // Load alpha.
" ld1 {v7.h}[0],[x8]                          \n\t" // Load beta
"                                            \n\t"
//" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
//" bne .SGENSTORED                            \n\t"
"                                            \n\t"
" .SCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
"                                            \n\t"
" dup  v0.8h, wzr                            \n\t"
" dup  v1.8h, wzr                            \n\t"
" dup  v2.8h, wzr                            \n\t"
" dup  v3.8h, wzr                            \n\t"
" dup  v4.8h, wzr                            \n\t"
" dup  v5.8h, wzr                            \n\t"
"                                            \n\t"
" cmp x7,#0                               \n\t"
" beq .SBETAZEROCOLSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v0.8h,v1.8h, v2.8h}, [x2]             \n\t" //Load column 0 of C
" ld1 {v3.8h, v4.8h, v5.8h}, [x16]           \n\t" //Load column 1 of C
"                                            \n\t"
" mul v0.8h,v0.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v1.8h,v1.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v2.8h,v2.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v3.8h,v3.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v4.8h,v4.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v5.8h,v5.8h,v7.h[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" mla v0.8h,v8.8h,v7.h[1]                   \n\t" // Scale by alpha
" mla v1.8h,v9.8h,v7.h[1]                   \n\t" // Scale by alpha
" mla v2.8h,v10.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v3.8h,v11.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v4.8h,v12.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v5.8h,v13.8h,v7.h[1]                  \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v0.8h, v1.8h, v2.8h}, [x2]                          \n\t" //Store column 0 of C
" st1 {v3.8h, v4.8h, v5.8h}, [x16]                         \n\t" //Store column 1 of C
"                                            \n\t"
" dup  v8.8h, wzr                            \n\t"
" dup  v9.8h, wzr                            \n\t"
" dup  v10.8h, wzr                           \n\t"
" dup  v11.8h, wzr                           \n\t"
" dup  v12.8h, wzr                           \n\t"
" dup  v13.8h, wzr                           \n\t"
" dup  v0.8h, wzr                            \n\t"
" dup  v1.8h, wzr                            \n\t"
" dup  v2.8h, wzr                            \n\t"
" dup  v3.8h, wzr                            \n\t"
" dup  v4.8h, wzr                            \n\t"
" dup  v5.8h, wzr                            \n\t"

"                                            \n\t"
" cmp x7,#0                               \n\t"
" beq .SBETAZEROCOLSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v8.8h, v9.8h, v10.8h}, [x17]          \n\t" //Load column 2 of C
" ld1 {v11.8h, v12.8h, v13.8h}, [x18]        \n\t" //Load column 3 of C
" ld1 {v0.8h, v1.8h, v2.8h}, [x19]           \n\t" //Load column 4 of C
" ld1 {v3.8h, v4.8h, v5.8h}, [x20]           \n\t" //Load column 5 of C
"                                            \n\t"
" mul v8.8h, v8.8h, v7.h[0]                 \n\t" // Scale by beta
" mul v9.8h, v9.8h, v7.h[0]                 \n\t" // Scale by beta
" mul v10.8h,v10.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v11.8h,v11.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v12.8h,v12.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v13.8h,v13.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v0.8h,v0.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v1.8h,v1.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v2.8h,v2.8h,v7.h[0]                   \n\t" // Scale by beta
" mul v3.8h,v3.8h,v7.h[0]                   \n\t" // Scale by beta"
" mul v4.8h,v4.8h,v7.h[0]                   \n\t" // Scale by beta"
" mul v5.8h,v5.8h,v7.h[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" mla v8.8h,v14.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v9.8h, v15.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v10.8h, v16.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v11.8h,v17.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v12.8h,v18.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v13.8h,v19.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v0.8h,v20.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v1.8h,v21.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v2.8h,v22.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v3.8h,v23.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v4.8h,v24.8h,v7.h[1]                  \n\t" // Scale by alpha
" mla v5.8h,v25.8h,v7.h[1]                  \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v8.8h, v9.8h, v10.8h}, [x17]          \n\t" //Store column 2 of C
" st1 {v11.8h, v12.8h, v13.8h}, [x18]        \n\t" //Store column 3 of C
" st1 {v0.8h, v1.8h, v2.8h}, [x19]           \n\t" //Store column 4 of C
" st1 {v3.8h, v4.8h, v5.8h}, [x20]           \n\t" //Store column 5 of C
"                                            \n\t"
" dup  v14.8h, wzr                           \n\t"
" dup  v15.8h, wzr                           \n\t"
" dup  v16.8h, wzr                           \n\t"
" dup  v17.8h, wzr                           \n\t"
" dup  v18.8h, wzr                           \n\t"
" dup  v19.8h, wzr                           \n\t"
"                                            \n\t"
" cmp x7,#0                              \n\t"
" beq .SBETAZEROCOLSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v14.8h, v15.8h, v16.8h}, [x21]        \n\t" //Load column 6 of C
" ld1 {v17.8h, v18.8h, v19.8h}, [x22]        \n\t" //Load column 7 of C
"                                            \n\t"
" mul v14.8h,v14.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v15.8h,v15.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v16.8h,v16.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v17.8h,v17.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v18.8h,v18.8h,v7.h[0]                 \n\t" // Scale by beta
" mul v19.8h,v19.8h,v7.h[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS3:                     \n\t"
"                                            \n\t"
//" prfm pldl2keep,[x3]                        \n\t"
//" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" mla v14.8h,v26.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v15.8h,v27.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v16.8h,v28.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v17.8h,v29.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v18.8h,v30.8h,v7.h[1]                 \n\t" // Scale by alpha
" mla v19.8h,v31.8h,v7.h[1]                 \n\t" // Scale by alpha

"                                            \n\t"
" st1 {v14.8h, v15.8h, v16.8h}, [x21]        \n\t" //Store column 6 of C
" st1 {v17.8h, v18.8h, v19.8h}, [x22]        \n\t" //Store column 7 of C
"                                            \n\t"
" .SEND:                                     \n\t" // Done!
"                                            \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_iter] "m" (k_iter), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 7
 [cs_c]   "m" (cs_c)/*,   // 8
 [a_next] "m" (a_next), // 9
 [b_next] "m" (b_next) // 10*/
:// Register clobber list
 "x0", "x1", "x2","x3","x4",
 "x5", "x6", "x7", "x8",
 "x9", "x10","x11","x12",
 "x13","x14","x15",
 "x16","x17","x18","x19",       
 "x20","x21","x22",
 "v0", "v1", "v2", "v3",
 "v4", "v5", "v6", "v7",
 "v8", "v9", "v10","v11",
 "v12","v13","v14","v15",
 "v16","v17","v18","v19",
 "v20","v21","v22","v23",
 "v24","v25","v26","v27",
 "v28","v29","v30","v31"
);

}

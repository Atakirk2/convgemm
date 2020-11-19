#include <blis.h>

void hgemm_armv8a_asm_8x24
     (
       dim_t               k0,
       _Float16*     restrict alpha,
       _Float16*     restrict a,
       _Float16*     restrict b,
       _Float16*     restrict beta,
       _Float16*     restrict c, inc_t rs_c0, inc_t cs_c0//,
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
" add x23,x22,x10                            \n\t" //Load address Column 8 of C
" add x24,x23,x10                            \n\t" //Load address Column 9 of C
" add x25,x24,x10                            \n\t" //Load address Column 10 of C
" add x26,x25,x10                            \n\t" //Load address Column 11 of C
" add x27,x26,x10                            \n\t" //Load address Column 12 of C
" add x28,x27,x10                            \n\t" //Load address Column 13 of C
" add x11,x28,x10                            \n\t" //Load address Column 14 of C
" add x12,x11,x10                            \n\t" //Load address Column 15 of C
" add x9,x12,x10                            \n\t" //Load address Column 16 of C
" add x13,x9,x10                            \n\t" //Load address Column 17 of C
"                                            \n\t"
" ld1 {v0.8h}, [x0]                   \n\t" //Load a
"                                            \n\t"
" ld1 {v2.8h, v3.8h, v4.8h}, [x1]            \n\t" //Load b
"                                            \n\t"
" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x16]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x17]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x18]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x19]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x25]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x26]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x27]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x28]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x11]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x12]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x9]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x13]                       \n\t" // Prefetch c.
"                                            \n\t"
" dup  v8.8h, wzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #192]              \n\t" 
" dup  v9.8h, wzr                            \n\t" // Vector for accummulating column 1
" prfm    PLDL1KEEP, [x1, #256]              \n\t"
" dup  v10.8h, wzr                           \n\t" // Vector for accummulating column 2
" prfm    PLDL1KEEP, [x1, #320]              \n\t"
" dup  v11.8h, wzr                           \n\t" // Vector for accummulating column 3
" dup  v12.8h, wzr                           \n\t" // Vector for accummulating column 4 
" dup  v13.8h, wzr                           \n\t" // Vector for accummulating column 5
"                                            \n\t"
" dup  v14.8h, wzr                           \n\t" // Vector for accummulating column 6
" prfm    PLDL1KEEP, [x0, #128]              \n\t"
" dup  v15.8h, wzr                           \n\t" // Vector for accummulating column 7
" prfm    PLDL1KEEP, [x0, #192]              \n\t"
" dup  v16.8h, wzr                           \n\t" // Vector for accummulating column 8
" dup  v17.8h, wzr                           \n\t" // Vector for accummulating column 9
" dup  v18.8h, wzr                           \n\t" // Vector for accummulating column 10 
" dup  v19.8h, wzr                           \n\t" // Vector for accummulating column 11
"                                            \n\t"
" dup  v20.8h, wzr                           \n\t" // Vector for accummulating column 12 
" dup  v21.8h, wzr                           \n\t" // Vector for accummulating column 13
" dup  v22.8h, wzr                           \n\t" // Vector for accummulating column 14
" dup  v23.8h, wzr                           \n\t" // Vector for accummulating column 15
" dup  v24.8h, wzr                           \n\t" // Vector for accummulating column 16 
" dup  v25.8h, wzr                           \n\t" // Vector for accummulating column 17
"                                            \n\t"
" dup  v26.8h, wzr                           \n\t" // Vector for accummulating column 18 
" dup  v27.8h, wzr                           \n\t" // Vector for accummulating column 19
" dup  v28.8h, wzr                           \n\t" // Vector for accummulating column 20
" dup  v29.8h, wzr                           \n\t" // Vector for accummulating column 21
" dup  v30.8h, wzr                           \n\t" // Vector for accummulating column 22 
" dup  v31.8h, wzr                           \n\t" // Vector for accummulating column 23
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .SCONSIDERKLEFT                        \n\t"
"                                            \n\t"
"add x0, x0, #16                             \n\t" //update address of A
"add x1, x1, #48                             \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .SLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" .SLOOPKITER:                               \n\t" // Body of the k_iter loop.
"                                            \n\t"
" ld1 {v1.8h}, [x0],#16                      \n\t"
" fmla v8.8h, v0.8h,v2.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v0.8h,v2.h[1]                  \n\t" // Accummulate.
" ld1 {v5.8h, v6.8h, v7.8h}, [x1],#48        \n\t"
" fmla v10.8h,v0.8h,v2.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v0.8h,v2.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v0.8h,v2.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v0.8h,v2.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v0.8h,v2.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v0.8h,v2.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v0.8h,v3.h[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #288]              \n\t" 
" fmla v17.8h,v0.8h,v3.h[1]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #352]              \n\t" 
" fmla v18.8h,v0.8h,v3.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v0.8h,v3.h[3]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #416]              \n\t" 
" fmla v20.8h,v0.8h,v3.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v0.8h,v3.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v0.8h,v3.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v0.8h,v3.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v0.8h,v4.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v0.8h,v4.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v0.8h,v4.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v0.8h,v4.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v0.8h,v4.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v0.8h,v4.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v0.8h,v4.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v0.8h,v4.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" ld1 {v0.8h}, [x0],#16                      \n\t"
" fmla v8.8h, v1.8h,v5.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v1.8h,v5.h[1]                  \n\t" // Accummulate.
" ld1 {v2.8h, v3.8h, v4.8h}, [x1],#48        \n\t"
" fmla v10.8h,v1.8h,v5.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v1.8h,v5.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v1.8h,v5.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v1.8h,v5.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v1.8h,v5.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v1.8h,v5.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v1.8h,v6.h[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #192]              \n\t"
" fmla v17.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #256]              \n\t"
" fmla v18.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v1.8h,v7.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v1.8h,v7.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v1.8h,v7.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v1.8h,v7.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v1.8h,v7.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v1.8h,v7.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v1.8h,v7.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v1.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" ld1 {v1.8h}, [x0],#16                      \n\t"
" fmla v8.8h, v0.8h,v2.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v0.8h,v2.h[1]                  \n\t" // Accummulate.
" ld1 {v5.8h, v6.8h, v7.8h}, [x1],#48        \n\t"
" fmla v10.8h,v0.8h,v2.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v0.8h,v2.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v0.8h,v2.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v0.8h,v2.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v0.8h,v2.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v0.8h,v2.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v0.8h,v3.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v0.8h,v3.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v0.8h,v3.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v0.8h,v3.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v0.8h,v3.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v0.8h,v3.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v0.8h,v3.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v0.8h,v3.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v0.8h,v4.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v0.8h,v4.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v0.8h,v4.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v0.8h,v4.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v0.8h,v4.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v0.8h,v4.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v0.8h,v4.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v0.8h,v4.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" ld1 {v0.8h}, [x0],#16                      \n\t"
" fmla v8.8h, v1.8h,v5.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v1.8h,v5.h[1]                  \n\t" // Accummulate.
" ld1 {v2.8h, v3.8h, v4.8h}, [x1],#48        \n\t"
" fmla v10.8h,v1.8h,v5.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v1.8h,v5.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v1.8h,v5.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v1.8h,v5.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v1.8h,v5.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v1.8h,v5.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v1.8h,v6.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v1.8h,v7.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v1.8h,v7.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v1.8h,v7.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v1.8h,v7.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v1.8h,v7.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v1.8h,v7.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v1.8h,v7.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v1.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 4
" sub x5,x5,1                                \n\t" // i-=1.
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .SLOOPKITER                            \n\t"
"                                            \n\t" 
" .SLASTITER:                                \n\t" // Last iteration of k_iter loop.
"                                            \n\t" 
" ld1 {v1.8h}, [x0],#16                      \n\t"
" fmla v8.8h, v0.8h,v2.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v0.8h,v2.h[1]                  \n\t" // Accummulate.
" ld1 {v5.8h, v6.8h, v7.8h}, [x1],#48        \n\t"
" fmla v10.8h,v0.8h,v2.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v0.8h,v2.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v0.8h,v2.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v0.8h,v2.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v0.8h,v2.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v0.8h,v2.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v0.8h,v3.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v0.8h,v3.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v0.8h,v3.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v0.8h,v3.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v0.8h,v3.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v0.8h,v3.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v0.8h,v3.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v0.8h,v3.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v0.8h,v4.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v0.8h,v4.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v0.8h,v4.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v0.8h,v4.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v0.8h,v4.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v0.8h,v4.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v0.8h,v4.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v0.8h,v4.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" ld1 {v0.8h}, [x0],#16                      \n\t"
" fmla v8.8h, v1.8h,v5.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v1.8h,v5.h[1]                  \n\t" // Accummulate.
" ld1 {v2.8h, v3.8h, v4.8h}, [x1],#48        \n\t"
" fmla v10.8h,v1.8h,v5.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v1.8h,v5.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v1.8h,v5.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v1.8h,v5.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v1.8h,v5.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v1.8h,v5.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v1.8h,v6.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v1.8h,v7.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v1.8h,v7.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v1.8h,v7.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v1.8h,v7.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v1.8h,v7.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v1.8h,v7.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v1.8h,v7.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v1.8h,v7.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" ld1 {v1.8h}, [x0],#16                      \n\t"
" ld1 {v5.8h, v6.8h, v7.8h}, [x1],#48        \n\t"
" fmla v8.8h, v0.8h,v2.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v0.8h,v2.h[1]                  \n\t" // Accummulate.
" fmla v10.8h,v0.8h,v2.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v0.8h,v2.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v0.8h,v2.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v0.8h,v2.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v0.8h,v2.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v0.8h,v2.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v0.8h,v3.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v0.8h,v3.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v0.8h,v3.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v0.8h,v3.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v0.8h,v3.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v0.8h,v3.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v0.8h,v3.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v0.8h,v3.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v0.8h,v4.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v0.8h,v4.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v0.8h,v4.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v0.8h,v4.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v0.8h,v4.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v0.8h,v4.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v0.8h,v4.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v0.8h,v4.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" fmla v8.8h, v1.8h,v5.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v1.8h,v5.h[1]                  \n\t" // Accummulate.
" fmla v10.8h,v1.8h,v5.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v1.8h,v5.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v1.8h,v5.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v1.8h,v5.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v1.8h,v5.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v1.8h,v5.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v1.8h,v6.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v1.8h,v6.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v1.8h,v6.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v1.8h,v6.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v1.8h,v6.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v1.8h,v6.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v1.8h,v6.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v1.8h,v6.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v1.8h,v7.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v1.8h,v7.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v1.8h,v7.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v1.8h,v7.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v1.8h,v7.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v1.8h,v7.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v1.8h,v7.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v1.8h,v7.h[7]                  \n\t" // Accummulate.
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
" ld1 {v0.8h}, [x0],#16                      \n\t"
" ld1 {v2.8h, v3.8h, v4.8h}, [x1],#48        \n\t"
" fmla v8.8h, v0.8h,v2.h[0]                  \n\t" // Accummulate.
" fmla v9.8h, v0.8h,v2.h[1]                  \n\t" // Accummulate.
" fmla v10.8h,v0.8h,v2.h[2]                  \n\t" // Accummulate.
" fmla v11.8h,v0.8h,v2.h[3]                  \n\t" // Accummulate.
" fmla v12.8h,v0.8h,v2.h[4]                  \n\t" // Accummulate.
" fmla v13.8h,v0.8h,v2.h[5]                  \n\t" // Accummulate.
" fmla v14.8h,v0.8h,v2.h[6]                  \n\t" // Accummulate.
" fmla v15.8h,v0.8h,v2.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.8h,v0.8h,v3.h[0]                  \n\t" // Accummulate.
" fmla v17.8h,v0.8h,v3.h[1]                  \n\t" // Accummulate.
" fmla v18.8h,v0.8h,v3.h[2]                  \n\t" // Accummulate.
" fmla v19.8h,v0.8h,v3.h[3]                  \n\t" // Accummulate.
" fmla v20.8h,v0.8h,v3.h[4]                  \n\t" // Accummulate.
" fmla v21.8h,v0.8h,v3.h[5]                  \n\t" // Accummulate.
" fmla v22.8h,v0.8h,v3.h[6]                  \n\t" // Accummulate.
" fmla v23.8h,v0.8h,v3.h[7]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.8h,v0.8h,v4.h[0]                  \n\t" // Accummulate.
" fmla v25.8h,v0.8h,v4.h[1]                  \n\t" // Accummulate.
" fmla v26.8h,v0.8h,v4.h[2]                  \n\t" // Accummulate.
" fmla v27.8h,v0.8h,v4.h[3]                  \n\t" // Accummulate.
" fmla v28.8h,v0.8h,v4.h[4]                  \n\t" // Accummulate.
" fmla v29.8h,v0.8h,v4.h[5]                  \n\t" // Accummulate.
" fmla v30.8h,v0.8h,v4.h[6]                  \n\t" // Accummulate.
" fmla v31.8h,v0.8h,v4.h[7]                  \n\t" // Accummulate.
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
" add x5,x13,x10                             \n\t" //Load address Column 18 of C
" add x6,x5,x10                              \n\t" //Load address Column 19 of C
" add x0,x6,x10                              \n\t" //Load address Column 20 of C
" add x1,x0,x10                              \n\t" //Load address Column 21 of C
" add x7,x1,x10                              \n\t" //Load address Column 22 of C
" add x8,x7,x10                              \n\t" //Load address Column 23 of C
"                                            \n\t"
" prfm pldl1keep,[x5]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x6]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x0]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x1]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x7]                        \n\t" // Prefetch c.
" prfm pldl1keep,[x8]                        \n\t" // Prefetch c.
"                                            \n\t"
" dup  v0.8h, wzr                            \n\t"
" dup  v1.8h, wzr                            \n\t"
" dup  v2.8h, wzr                            \n\t"
" dup  v3.8h, wzr                            \n\t"
" dup  v4.8h, wzr                            \n\t"
" dup  v5.8h, wzr                            \n\t"
" dup  v6.8h, wzr                            \n\t"
"                                            \n\t"
" fcmp h7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v0.8h}, [x2]                          \n\t" //Load column 0 of C
" ld1 {v1.8h}, [x16]                         \n\t" //Load column 1 of C
" ld1 {v2.8h}, [x17]                         \n\t" //Load column 2 of C
" ld1 {v3.8h}, [x18]                         \n\t" //Load column 3 of C
" ld1 {v4.8h}, [x19]                         \n\t" //Load column 4 of C
"                                            \n\t"
" fmul v0.8h,v0.8h,v7.h[0]                   \n\t" // Scale by beta
" ld1 {v5.8h}, [x20]                         \n\t" //Load column 5 of C
" fmul v1.8h,v1.8h,v7.h[0]                   \n\t" // Scale by beta
" ld1 {v6.8h}, [x21]                         \n\t" //Load column 6 of C
" fmul v2.8h,v2.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v3.8h,v3.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v4.8h,v4.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v5.8h,v5.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v6.8h,v6.8h,v7.h[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla v0.8h,v8.8h,v7.h[1]                   \n\t" // Scale by alpha
" fmla v1.8h,v9.8h,v7.h[1]                   \n\t" // Scale by alpha
" fmla v2.8h,v10.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v3.8h,v11.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v4.8h,v12.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v5.8h,v13.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v6.8h,v14.8h,v7.h[1]                  \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v0.8h}, [x2]                          \n\t" //Store column 0 of C
" st1 {v1.8h}, [x16]                         \n\t" //Store column 1 of C
" st1 {v2.8h}, [x17]                         \n\t" //Store column 2 of C
" st1 {v3.8h}, [x18]                         \n\t" //Store column 3 of C
" st1 {v4.8h}, [x19]                         \n\t" //Store column 4 of C
" st1 {v5.8h}, [x20]                         \n\t" //Store column 5 of C
" st1 {v6.8h}, [x21]                         \n\t" //Store column 6 of C
"                                            \n\t"
" dup  v8.8h, wzr                            \n\t"
" dup  v9.8h, wzr                            \n\t"
" dup  v10.8h, wzr                           \n\t"
" dup  v11.8h, wzr                           \n\t"
" dup  v12.8h, wzr                           \n\t"
" dup  v13.8h, wzr                           \n\t"
" dup  v14.8h, wzr                           \n\t"
" dup  v0.8h, wzr                            \n\t"
" dup  v1.8h, wzr                            \n\t"
" dup  v2.8h, wzr                            \n\t"
" dup  v3.8h, wzr                            \n\t"
" dup  v4.8h, wzr                            \n\t"
" dup  v5.8h, wzr                            \n\t"
" dup  v6.8h, wzr                            \n\t"
"                                            \n\t"
" fcmp h7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v8.8h}, [x22]                          \n\t" //Load column 7 of C
" ld1 {v9.8h}, [x23]                         \n\t" //Load column 8 of C
" ld1 {v10.8h}, [x24]                         \n\t" //Load column 9 of C
" ld1 {v11.8h}, [x25]                         \n\t" //Load column 10 of C
" ld1 {v12.8h}, [x26]                         \n\t" //Load column 11 of C
" ld1 {v13.8h}, [x27]                          \n\t" //Load column 12 of C
" ld1 {v14.8h}, [x28]                         \n\t" //Load column 13 of C
" ld1 {v0.8h}, [x11]                         \n\t" //Load column 14 of C
" ld1 {v1.8h}, [x12]                         \n\t" //Load column 15 of C
" ld1 {v2.8h}, [x9]                         \n\t" //Load column 16 of C
" ld1 {v3.8h}, [x13]                         \n\t" //Load column 17 of C
" ld1 {v4.8h}, [x5]                         \n\t" //Load column 18 of C
" ld1 {v5.8h}, [x6]                         \n\t" //Load column 19 of C
" ld1 {v6.8h}, [x0]                         \n\t" //Load column 20 of C
"                                            \n\t"
" fmul v8.8h, v8.8h, v7.h[0]                 \n\t" // Scale by beta
" fmul v9.8h, v9.8h, v7.h[0]                 \n\t" // Scale by beta
" fmul v10.8h,v10.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v11.8h,v11.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v12.8h,v12.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v13.8h,v13.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v14.8h,v14.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v0.8h,v0.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v1.8h,v1.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v2.8h,v2.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v3.8h,v3.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v4.8h,v4.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v5.8h,v5.8h,v7.h[0]                   \n\t" // Scale by beta
" fmul v6.8h,v6.8h,v7.h[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla v8.8h, v15.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v9.8h, v16.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v10.8h,v17.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v11.8h,v18.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v12.8h,v19.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v13.8h,v20.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v14.8h,v21.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v0.8h,v22.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v1.8h,v23.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v2.8h,v24.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v3.8h,v25.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v4.8h,v26.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v5.8h,v27.8h,v7.h[1]                  \n\t" // Scale by alpha
" fmla v6.8h,v28.8h,v7.h[1]                  \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v8.8h}, [x22]                         \n\t" //Store column 7 of C
" st1 {v9.8h}, [x23]                         \n\t" //Store column 8 of C
" st1 {v10.8h}, [x24]                        \n\t" //Store column 9 of C
" st1 {v11.8h}, [x25]                        \n\t" //Store column 10 of C
" st1 {v12.8h}, [x26]                        \n\t" //Store column 11 of C
" st1 {v13.8h}, [x27]                        \n\t" //Store column 12 of C
" st1 {v14.8h}, [x28]                        \n\t" //Store column 13 of C
" st1 {v0.8h}, [x11]                         \n\t" //Store column 14 of C
" st1 {v1.8h}, [x12]                         \n\t" //Store column 15 of C
" st1 {v2.8h}, [x9]                          \n\t" //Store column 16 of C
" st1 {v3.8h}, [x13]                         \n\t" //Store column 17 of C
" st1 {v4.8h}, [x5]                          \n\t" //Store column 18 of C
" st1 {v5.8h}, [x6]                          \n\t" //Store column 19 of C
" st1 {v6.8h}, [x0]                          \n\t" //Store column 20 of C
"                                            \n\t"
" dup  v15.8h, wzr                           \n\t"
" dup  v16.8h, wzr                           \n\t"
" dup  v17.8h, wzr                           \n\t"
"                                            \n\t"
" fcmp h7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v15.8h}, [x1]                         \n\t" //Load column 21 of C
" ld1 {v16.8h}, [x7]                         \n\t" //Load column 22 of C
" ld1 {v17.8h}, [x8]                         \n\t" //Load column 23 of C
"                                            \n\t"
" fmul v15.8h,v15.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v16.8h,v16.8h,v7.h[0]                 \n\t" // Scale by beta
" fmul v17.8h,v17.8h,v7.h[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS3:                     \n\t"
"                                            \n\t"
//" prfm pldl2keep,[x3]                        \n\t"
//" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" fmla v15.8h,v29.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v16.8h,v30.8h,v7.h[1]                 \n\t" // Scale by alpha
" fmla v17.8h,v31.8h,v7.h[1]                 \n\t" // Scale by alpha

"                                            \n\t"
" st1 {v15.8h}, [x1]                         \n\t" //Store column 21 of C
" st1 {v16.8h}, [x7]                         \n\t" //Store column 22 of C
" st1 {v17.8h}, [x8]                         \n\t" //Store column 23 of C
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
 "x20","x21","x22","x23",
 "x24","x25","x26","x27","x28",
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

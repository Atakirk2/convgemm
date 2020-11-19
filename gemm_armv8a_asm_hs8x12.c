#include <blis.h>

void hsgemm_armv8a_asm_8x12
     (
       dim_t               k0,
       _Float16*     restrict alpha,
       _Float16*     restrict a,
       _Float16*     restrict b,
       _Float16*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0//,
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
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
"                                            \n\t"
" lsl x10,x9,#2                              \n\t" // cs_c * sizeof(float) -- AUX.
" lsl x14,x13,#2                             \n\t" // rs_c * sizeof(float).
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
"                                            \n\t"
" ld1 {v0.4h, v1.4h}, [x0]                   \n\t"
" fcvtl v0.4s, v0.4h                          \n\t"
" fcvtl v1.4s, v1.4h                          \n\t"
//" ldr q0, [x0]                               \n\t"
//" ldr q1, [x0, #16]                          \n\t" // Load a
"                                            \n\t"
" ld1 {v2.4h, v3.4h, v4.4h}, [x1]              \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
//" ldr q2, [x1]                               \n\t" // Load b
//" ldr q3, [x1, #16]                          \n\t"
//" ldr q4, [x1, #32]                          \n\t"
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
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #192]              \n\t" 
" dup  v9.4s, wzr                            \n\t" // Vector for accummulating column 0
" prfm    PLDL1KEEP, [x1, #256]              \n\t"
" dup  v10.4s, wzr                           \n\t" // Vector for accummulating column 1
" prfm    PLDL1KEEP, [x1, #320]              \n\t"
" dup  v11.4s, wzr                           \n\t" // Vector for accummulating column 1
" dup  v12.4s, wzr                           \n\t" // Vector for accummulating column 2 
" dup  v13.4s, wzr                           \n\t" // Vector for accummulating column 2
"                                            \n\t"
" dup  v14.4s, wzr                           \n\t" // Vector for accummulating column 3
" prfm    PLDL1KEEP, [x0, #128]              \n\t"
" dup  v15.4s, wzr                           \n\t" // Vector for accummulating column 3
" prfm    PLDL1KEEP, [x0, #192]              \n\t"
" dup  v16.4s, wzr                           \n\t" // Vector for accummulating column 4
" dup  v17.4s, wzr                           \n\t" // Vector for accummulating column 4
" dup  v18.4s, wzr                           \n\t" // Vector for accummulating column 5 
" dup  v19.4s, wzr                           \n\t" // Vector for accummulating column 5
"                                            \n\t"
" dup  v20.4s, wzr                           \n\t" // Vector for accummulating column 6 
" dup  v21.4s, wzr                           \n\t" // Vector for accummulating column 6
" dup  v22.4s, wzr                           \n\t" // Vector for accummulating column 7
" dup  v23.4s, wzr                           \n\t" // Vector for accummulating column 7
" dup  v24.4s, wzr                           \n\t" // Vector for accummulating column 8 
" dup  v25.4s, wzr                           \n\t" // Vector for accummulating column 8
"                                            \n\t"
" dup  v26.4s, wzr                           \n\t" // Vector for accummulating column 9 
" dup  v27.4s, wzr                           \n\t" // Vector for accummulating column 9
" dup  v28.4s, wzr                           \n\t" // Vector for accummulating column 10
" dup  v29.4s, wzr                           \n\t" // Vector for accummulating column 10
" dup  v30.4s, wzr                           \n\t" // Vector for accummulating column 11 
" dup  v31.4s, wzr                           \n\t" // Vector for accummulating column 11
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .SCONSIDERKLEFT                        \n\t"
"                                            \n\t"
"add x0, x0, #16                             \n\t" //update address of A
"add x1, x1, #24                             \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .SLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" .SLOOPKITER:                               \n\t" // Body of the k_iter loop.
"                                            \n\t"
" ld1 {v5.4h, v6.4h}, [x0],#16                     \n\t"
" fcvtl v5.4s, v5.4h                          \n\t"
" fcvtl v6.4s, v6.4h                          \n\t"

" fmla v8.4s, v0.4s,v2.s[0]                  \n\t" // Accummulate.
" fmla v9.4s, v1.4s,v2.s[0]                  \n\t" // Accummulate.
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #336]              \n\t" 
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #400]              \n\t" 
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x1, #464]              \n\t" 
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1]                   \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
//" ldr q4, [x1, #32]                          \n\t"
" ldr d4, [x1, #16]                          \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
//" ld1 {v4.4s}, [x1]                          \n\t"
" add x1, x1, #24                           \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" ld1 {v0.4h, v1.4h}, [x0],#16                   \n\t"
" fcvtl v0.4s, v0.4h                          \n\t"
" fcvtl v1.4s, v1.4h                          \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #224]              \n\t"
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" prfm    PLDL1KEEP, [x0, #288]              \n\t"
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1]              \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
//" ldr q4, [x1, #80]                          \n\t"
" ldr d4, [x1, #16]                          \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
//" ld1 {v4.4s}, [x1]                          \n\t"
" add x1, x1, #24                           \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" ld1 {v5.4h, v6.4h}, [x0],#16                   \n\t"
" fcvtl v5.4s, v5.4h                          \n\t"
" fcvtl v6.4s, v6.4h                          \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1]              \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
//" ldr q4, [x1, #128]                         \n\t"
" ldr d4, [x1, #16]                          \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
//" ld1 {v4.4s}, [x1]                          \n\t"
" add x1, x1, #24                           \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" ld1 {v0.4h, v1.4h}, [x0], #16              \n\t"
" fcvtl v0.4s, v0.4h                          \n\t"
" fcvtl v1.4s, v1.4h                          \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1]              \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
//" ldr q4, [x1, #176]                         \n\t"
" ldr d4, [x1, #16]                          \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
//" ld1 {v4.4s}, [x1]                          \n\t"
" add x1, x1, #24                           \n\t"
//" add x1, x1, #192                           \n\t"
//" add x0, x0, #128                            \n\t" 
"                                            \n\t" //End It 4
" sub x5,x5,1                                \n\t" // i-=1.
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
"                                            \n\t"
" bne .SLOOPKITER                            \n\t"
"                                            \n\t" 
" .SLASTITER:                                \n\t" // Last iteration of k_iter loop.
"                                            \n\t" 
"                                            \n\t"
" ld1 {v5.4h, v6.4h}, [x0],#16                     \n\t"
" fcvtl v5.4s, v5.4h                          \n\t"
" fcvtl v6.4s, v6.4h                          \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1],#16                   \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" ld1 {v4.4h}, [x1], #8                      \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"

"                                            \n\t" //End It 1
"                                            \n\t"
" ld1 {v0.4h, v1.4h}, [x0],#16                   \n\t"
" fcvtl v0.4s, v0.4h                          \n\t"
" fcvtl v1.4s, v1.4h                          \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1],#16                   \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v4.4h}, [x1], #8                      \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
"                                            \n\t" //End It 2
"                                            \n\t"
" ld1 {v5.4h, v6.4h}, [x0],#16                     \n\t"
" fcvtl v5.4s, v5.4h                          \n\t"
" fcvtl v6.4s, v6.4h                          \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v2.4h, v3.4h}, [x1],#16                   \n\t"
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
"                                            \n\t"
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
" ld1 {v4.4h}, [x1], #8                      \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
//" add x1, x1, #72                           \n\t"
//" add x0, x0, #48                            \n\t"
"                                            \n\t" //End It 4
"                                            \n\t"
" .SCONSIDERKLEFT:                           \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .SPOSTACCUM                            \n\t" // else, we enter the k_left loop.
"                                            \n\t"
" .SLOOPKLEFT:                               \n\t" // Body of the left iterations
"                                            \n\t"
/*" ldr q0, [x0],#16                           \n\t"
" ldr q1, [x0],#16                           \n\t" // Load a
"                                            \n\t"
" ldr q2, [x1],#16                           \n\t" // Load b
" ldr q3, [x1],#16                           \n\t"
" ldr q4, [x1],#16                           \n\t"*/

" ld1 {v0.4h, v1.4h}, [x0] ,#16                  \n\t"//Load a
" fcvtl v0.4s, v0.4h                          \n\t"
" fcvtl v1.4s, v1.4h                          \n\t"
"                                            \n\t"
" ld1 {v2.4h, v3.4h, v4.4h}, [x1] ,#24             \n\t"//Load b
" fcvtl v2.4s, v2.4h                          \n\t"
" fcvtl v3.4s, v3.4h                          \n\t"
" fcvtl v4.4s, v4.4h                          \n\t"
"                                            \n\t"
" sub x6,x6,1                                \n\t" // i = i-1.
"                                            \n\t"
" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .SLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" .SPOSTACCUM:                               \n\t"
"                                            \n\t"
" ld1r {v6.4h},[x7]                          \n\t" // Load alpha.
" ld1r {v7.4h},[x8]                          \n\t" // Load beta
" fcvtl v6.4s, v6.4h                          \n\t"
" fcvtl v7.4s, v7.4h                          \n\t"
"                                            \n\t"
//" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
//" bne .SGENSTORED                            \n\t"
"                                            \n\t"
" .SCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
/*" ldr q0, [x2]                               \n\t" //Load column 0 of C
" ldr q1, [x2, #16]                          \n\t"
" ldr q2, [x16]                              \n\t" //Load column 1 of C
" ldr q3, [x16, #16]                         \n\t"
" ldr q4, [x17]                              \n\t" //Load column 2 of C
" ldr q5, [x17, #16]                         \n\t"*/
" ld1 {v0.4s,v1.4s}, [x2]                    \n\t" //Load column 0 of C
" ld1 {v2.4s,v3.4s}, [x16]                   \n\t" //Load column 1 of C
" ld1 {v4.4s,v5.4s}, [x17]                   \n\t" //Load column 2 of C
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla v0.4s,v8.4s,v6.s[0]                   \n\t" // Scale by alpha
" fmla v1.4s,v9.4s,v6.s[0]                   \n\t" // Scale by alpha
" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
/*" str q0, [x2]                               \n\t" //Store column 0 of C
" str q1, [x2, #16]                          \n\t"
" str q2, [x16]                              \n\t" //Store column 1 of C
" str q3, [x16, #16]                         \n\t"
" str q4, [x17]                              \n\t" //Store column 2 of C
" str q5, [x17, #16]                         \n\t"*/
" st1 {v0.4s,v1.4s}, [x2]                    \n\t" //Store column 0 of C
" st1 {v2.4s,v3.4s}, [x16]                   \n\t" //Store column 1 of C
" st1 {v4.4s,v5.4s}, [x17]                   \n\t" //Store column 2 of C
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                           \n\t"
" dup  v11.4s, wzr                           \n\t"
" dup  v12.4s, wzr                           \n\t"
" dup  v13.4s, wzr                           \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
/*" ldr q8, [x18]                              \n\t" //Load column 3 of C
" ldr q9, [x18, #16]                         \n\t"
" ldr q10, [x19]                             \n\t" //Load column 4 of C
" ldr q11, [x19, #16]                        \n\t"
" ldr q12, [x20]                             \n\t" //Load column 5 of C
" ldr q13, [x20, #16]                        \n\t"*/
" ld1 {v8.4s,v9.4s}, [x18]                    \n\t" //Load column 3 of C
" ld1 {v10.4s,v11.4s}, [x19]                   \n\t" //Load column 4 of C
" ld1 {v12.4s,v13.4s}, [x20]                   \n\t" //Load column 5 of C
" ld1 {v0.4s,v1.4s}, [x21]                    \n\t" //Load column 6 of C
" ld1 {v2.4s,v3.4s}, [x22]                   \n\t" //Load column 7 of C
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" ld1 {v4.4s,v5.4s}, [x23]                   \n\t" //Load column 8 of C
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
/*" str q8, [x18]                              \n\t" //Store column 3 of C
" str q9, [x18, #16]                         \n\t"
" str q10, [x19]                             \n\t" //Store column 4 of C
" str q11, [x19, #16]                        \n\t"
" str q12, [x20]                             \n\t" //Store column 5 of C
" str q13, [x20, #16]                        \n\t"*/
" st1 {v8.4s,v9.4s}, [x18]                    \n\t" //Store column 3 of C
" st1 {v10.4s,v11.4s}, [x19]                   \n\t" //Store column 4 of C
" st1 {v12.4s,v13.4s}, [x20]                   \n\t" //Store column 5 of C
" st1 {v0.4s,v1.4s}, [x21]                    \n\t" //Store column 6 of C
" st1 {v2.4s,v3.4s}, [x22]                   \n\t" //Store column 7 of C
" st1 {v4.4s,v5.4s}, [x23]                   \n\t" //Store column 8 of C
"                                            \n\t"
" dup  v14.4s, wzr                            \n\t"
" dup  v15.4s, wzr                            \n\t"
" dup  v16.4s, wzr                            \n\t"
" dup  v17.4s, wzr                            \n\t"
" dup  v18.4s, wzr                            \n\t"
" dup  v19.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS4                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
/*" ldr q8, [x24]                              \n\t" //Load column 9 of C
" ldr q9, [x24, #16]                         \n\t"
" ldr q10, [x25]                             \n\t" //Load column 10 of C
" ldr q11, [x25, #16]                        \n\t"
" ldr q12, [x26]                             \n\t" //Load column 11 of C
" ldr q13, [x26, #16]                        \n\t"*/
" ld1 {v14.4s,v15.4s}, [x24]                    \n\t" //Load column 9 of C
" ld1 {v16.4s,v17.4s}, [x25]                   \n\t" //Load column 10 of C
" ld1 {v18.4s,v19.4s}, [x26]                   \n\t" //Load column 11 of C
"                                            \n\t"
" fmul v14.4s,v14.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v15.4s,v15.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v16.4s,v16.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v17.4s,v17.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v18.4s,v18.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v19.4s,v19.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS4:                     \n\t"
"                                            \n\t"
//" prfm pldl2keep,[x3]                        \n\t"
//" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" fmla v14.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v15.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v16.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v17.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v18.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v19.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
/*" str q8, [x24]                              \n\t" //Store column 9 of C
" str q9, [x24, #16]                         \n\t"
" str q10, [x25]                             \n\t" //Store column 10 of C
" str q11, [x25, #16]                        \n\t"
" str q12, [x26]                             \n\t" //Store column 11 of C
" str q13, [x26, #16]                        \n\t"*/
" st1 {v14.4s,v15.4s}, [x24]                    \n\t" //Store column 6 of C
" st1 {v16.4s,v17.4s}, [x25]                   \n\t" //Store column 7 of C
" st1 {v18.4s,v19.4s}, [x26]                   \n\t" //Store column 8 of C
"                                            \n\t"
"                                            \n\t"
" b .SEND                                    \n\t" // Done (TODO: this obviously needs to be moved down to remove jump).
"                                            \n\t"
"                                            \n\t"
/*" .SGENSTORED:                               \n\t" // C is general-stride stored.
"                                            \n\t"
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x2                                \n\t"
"                                            \n\t"
" ld1 {v0.s}[0],[x27],x14                    \n\t" // Load c00  into quad and increment by rs_c.
" ld1 {v0.s}[1],[x27],x14                    \n\t" // Load c01  into quad and increment by rs_c.
" ld1 {v0.s}[2],[x27],x14                    \n\t" // Load c02  into quad and increment by rs_c.
" ld1 {v0.s}[3],[x27],x14                    \n\t" // Load c03  into quad and increment by rs_c.
" ld1 {v1.s}[0],[x27],x14                    \n\t" // Load c04  into quad and increment by rs_c.
" ld1 {v1.s}[1],[x27],x14                    \n\t" // Load c05  into quad and increment by rs_c.
" ld1 {v1.s}[2],[x27],x14                    \n\t" // Load c06  into quad and increment by rs_c.
" ld1 {v1.s}[3],[x27],x14                    \n\t" // Load c07  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x16                               \n\t"
"                                            \n\t"
" ld1 {v2.s}[0],[x27],x14                    \n\t" // Load c10  into quad and increment by rs_c.
" ld1 {v2.s}[1],[x27],x14                    \n\t" // Load c11  into quad and increment by rs_c.
" ld1 {v2.s}[2],[x27],x14                    \n\t" // Load c12  into quad and increment by rs_c.
" ld1 {v2.s}[3],[x27],x14                    \n\t" // Load c13  into quad and increment by rs_c.
" ld1 {v3.s}[0],[x27],x14                    \n\t" // Load c14  into quad and increment by rs_c.
" ld1 {v3.s}[1],[x27],x14                    \n\t" // Load c15  into quad and increment by rs_c.
" ld1 {v3.s}[2],[x27],x14                    \n\t" // Load c16  into quad and increment by rs_c.
" ld1 {v3.s}[3],[x27],x14                    \n\t" // Load c17  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x17                               \n\t"
"                                            \n\t"
" ld1 {v4.s}[0],[x27],x14                    \n\t" // Load c20  into quad and increment by rs_c.
" ld1 {v4.s}[1],[x27],x14                    \n\t" // Load c21  into quad and increment by rs_c.
" ld1 {v4.s}[2],[x27],x14                    \n\t" // Load c22  into quad and increment by rs_c.
" ld1 {v4.s}[3],[x27],x14                    \n\t" // Load c23  into quad and increment by rs_c.
" ld1 {v5.s}[0],[x27],x14                    \n\t" // Load c24  into quad and increment by rs_c.
" ld1 {v5.s}[1],[x27],x14                    \n\t" // Load c25  into quad and increment by rs_c.
" ld1 {v5.s}[2],[x27],x14                    \n\t" // Load c26  into quad and increment by rs_c.
" ld1 {v5.s}[3],[x27],x14                    \n\t" // Load c27  into quad and increment by rs_c.
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS1:                     \n\t"
"                                            \n\t"
" fmla v0.4s, v8.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v1.4s, v9.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x2                                \n\t"
"                                            \n\t"
" st1 {v0.s}[0],[x27],x14                    \n\t" // Store c00  into quad and increment by rs_c.
" st1 {v0.s}[1],[x27],x14                    \n\t" // Store c01  into quad and increment by rs_c.
" st1 {v0.s}[2],[x27],x14                    \n\t" // Store c02  into quad and increment by rs_c.
" st1 {v0.s}[3],[x27],x14                    \n\t" // Store c03  into quad and increment by rs_c.
" st1 {v1.s}[0],[x27],x14                    \n\t" // Store c04  into quad and increment by rs_c.
" st1 {v1.s}[1],[x27],x14                    \n\t" // Store c05  into quad and increment by rs_c.
" st1 {v1.s}[2],[x27],x14                    \n\t" // Store c06  into quad and increment by rs_c.
" st1 {v1.s}[3],[x27],x14                    \n\t" // Store c07  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x16                               \n\t"
"                                            \n\t"
" st1 {v2.s}[0],[x27],x14                    \n\t" // Store c10  into quad and increment by rs_c.
" st1 {v2.s}[1],[x27],x14                    \n\t" // Store c11  into quad and increment by rs_c.
" st1 {v2.s}[2],[x27],x14                    \n\t" // Store c12  into quad and increment by rs_c.
" st1 {v2.s}[3],[x27],x14                    \n\t" // Store c13  into quad and increment by rs_c.
" st1 {v3.s}[0],[x27],x14                    \n\t" // Store c14  into quad and increment by rs_c.
" st1 {v3.s}[1],[x27],x14                    \n\t" // Store c15  into quad and increment by rs_c.
" st1 {v3.s}[2],[x27],x14                    \n\t" // Store c16  into quad and increment by rs_c.
" st1 {v3.s}[3],[x27],x14                    \n\t" // Store c17  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x17                               \n\t"
"                                            \n\t"
" st1 {v4.s}[0],[x27],x14                    \n\t" // Store c20  into quad and increment by rs_c.
" st1 {v4.s}[1],[x27],x14                    \n\t" // Store c21  into quad and increment by rs_c.
" st1 {v4.s}[2],[x27],x14                    \n\t" // Store c22  into quad and increment by rs_c.
" st1 {v4.s}[3],[x27],x14                    \n\t" // Store c23  into quad and increment by rs_c.
" st1 {v5.s}[0],[x27],x14                    \n\t" // Store c24  into quad and increment by rs_c.
" st1 {v5.s}[1],[x27],x14                    \n\t" // Store c25  into quad and increment by rs_c.
" st1 {v5.s}[2],[x27],x14                    \n\t" // Store c26  into quad and increment by rs_c.
" st1 {v5.s}[3],[x27],x14                    \n\t" // Store c27  into quad and increment by rs_c.
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                           \n\t"
" dup  v11.4s, wzr                           \n\t"
" dup  v12.4s, wzr                           \n\t"
" dup  v13.4s, wzr                           \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x18                               \n\t"
"                                            \n\t"
" ld1 {v8.s}[0],[x27],x14                    \n\t" // Load c30  into quad and increment by rs_c.
" ld1 {v8.s}[1],[x27],x14                    \n\t" // Load c31  into quad and increment by rs_c.
" ld1 {v8.s}[2],[x27],x14                    \n\t" // Load c32  into quad and increment by rs_c.
" ld1 {v8.s}[3],[x27],x14                    \n\t" // Load c33  into quad and increment by rs_c.
" ld1 {v9.s}[0],[x27],x14                    \n\t" // Load c34  into quad and increment by rs_c.
" ld1 {v9.s}[1],[x27],x14                    \n\t" // Load c35  into quad and increment by rs_c.
" ld1 {v9.s}[2],[x27],x14                    \n\t" // Load c36  into quad and increment by rs_c.
" ld1 {v9.s}[3],[x27],x14                    \n\t" // Load c37  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x19                               \n\t"
"                                            \n\t"
" ld1 {v10.s}[0],[x27],x14                   \n\t" // Load c40  into quad and increment by rs_c.
" ld1 {v10.s}[1],[x27],x14                   \n\t" // Load c41  into quad and increment by rs_c.
" ld1 {v10.s}[2],[x27],x14                   \n\t" // Load c42  into quad and increment by rs_c.
" ld1 {v10.s}[3],[x27],x14                   \n\t" // Load c43  into quad and increment by rs_c.
" ld1 {v11.s}[0],[x27],x14                   \n\t" // Load c44  into quad and increment by rs_c.
" ld1 {v11.s}[1],[x27],x14                   \n\t" // Load c45  into quad and increment by rs_c.
" ld1 {v11.s}[2],[x27],x14                   \n\t" // Load c46  into quad and increment by rs_c.
" ld1 {v11.s}[3],[x27],x14                   \n\t" // Load c47  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x20                               \n\t"
"                                            \n\t"
" ld1 {v12.s}[0],[x27],x14                   \n\t" // Load c50  into quad and increment by rs_c.
" ld1 {v12.s}[1],[x27],x14                   \n\t" // Load c51  into quad and increment by rs_c.
" ld1 {v12.s}[2],[x27],x14                   \n\t" // Load c52  into quad and increment by rs_c.
" ld1 {v12.s}[3],[x27],x14                   \n\t" // Load c53  into quad and increment by rs_c.
" ld1 {v13.s}[0],[x27],x14                   \n\t" // Load c54  into quad and increment by rs_c.
" ld1 {v13.s}[1],[x27],x14                   \n\t" // Load c55  into quad and increment by rs_c.
" ld1 {v13.s}[2],[x27],x14                   \n\t" // Load c56  into quad and increment by rs_c.
" ld1 {v13.s}[3],[x27],x14                   \n\t" // Load c57  into quad and increment by rs_c.
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS2:                     \n\t"
"                                            \n\t"
" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x18                               \n\t"
"                                            \n\t"
" st1 {v8.s}[0],[x27],x14                    \n\t" // Store c30  into quad and increment by rs_c.
" st1 {v8.s}[1],[x27],x14                    \n\t" // Store c31  into quad and increment by rs_c.
" st1 {v8.s}[2],[x27],x14                    \n\t" // Store c32  into quad and increment by rs_c.
" st1 {v8.s}[3],[x27],x14                    \n\t" // Store c33  into quad and increment by rs_c.
" st1 {v9.s}[0],[x27],x14                    \n\t" // Store c34  into quad and increment by rs_c.
" st1 {v9.s}[1],[x27],x14                    \n\t" // Store c35  into quad and increment by rs_c.
" st1 {v9.s}[2],[x27],x14                    \n\t" // Store c36  into quad and increment by rs_c.
" st1 {v9.s}[3],[x27],x14                    \n\t" // Store c37  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x19                               \n\t"
"                                            \n\t"
" st1 {v10.s}[0],[x27],x14                   \n\t" // Store c40  into quad and increment by rs_c.
" st1 {v10.s}[1],[x27],x14                   \n\t" // Store c41  into quad and increment by rs_c.
" st1 {v10.s}[2],[x27],x14                   \n\t" // Store c42  into quad and increment by rs_c.
" st1 {v10.s}[3],[x27],x14                   \n\t" // Store c43  into quad and increment by rs_c.
" st1 {v11.s}[0],[x27],x14                   \n\t" // Store c44  into quad and increment by rs_c.
" st1 {v11.s}[1],[x27],x14                   \n\t" // Store c45  into quad and increment by rs_c.
" st1 {v11.s}[2],[x27],x14                   \n\t" // Store c46  into quad and increment by rs_c.
" st1 {v11.s}[3],[x27],x14                   \n\t" // Store c47  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x20                               \n\t"
"                                            \n\t"
" st1 {v12.s}[0],[x27],x14                   \n\t" // Store c50  into quad and increment by rs_c.
" st1 {v12.s}[1],[x27],x14                   \n\t" // Store c51  into quad and increment by rs_c.
" st1 {v12.s}[2],[x27],x14                   \n\t" // Store c52  into quad and increment by rs_c.
" st1 {v12.s}[3],[x27],x14                   \n\t" // Store c53  into quad and increment by rs_c.
" st1 {v13.s}[0],[x27],x14                   \n\t" // Store c54  into quad and increment by rs_c.
" st1 {v13.s}[1],[x27],x14                   \n\t" // Store c55  into quad and increment by rs_c.
" st1 {v13.s}[2],[x27],x14                   \n\t" // Store c56  into quad and increment by rs_c.
" st1 {v13.s}[3],[x27],x14                   \n\t" // Store c57  into quad and increment by rs_c.
"                                            \n\t"
" dup  v0.4s, wzr                            \n\t"
" dup  v1.4s, wzr                            \n\t"
" dup  v2.4s, wzr                            \n\t"
" dup  v3.4s, wzr                            \n\t"
" dup  v4.4s, wzr                            \n\t"
" dup  v5.4s, wzr                            \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS3                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x21                               \n\t"
"                                            \n\t"
" ld1 {v0.s}[0],[x27],x14                    \n\t" // Load c60  into quad and increment by rs_c.
" ld1 {v0.s}[1],[x27],x14                    \n\t" // Load c61  into quad and increment by rs_c.
" ld1 {v0.s}[2],[x27],x14                    \n\t" // Load c62  into quad and increment by rs_c.
" ld1 {v0.s}[3],[x27],x14                    \n\t" // Load c63  into quad and increment by rs_c.
" ld1 {v1.s}[0],[x27],x14                    \n\t" // Load c64  into quad and increment by rs_c.
" ld1 {v1.s}[1],[x27],x14                    \n\t" // Load c65  into quad and increment by rs_c.
" ld1 {v1.s}[2],[x27],x14                    \n\t" // Load c66  into quad and increment by rs_c.
" ld1 {v1.s}[3],[x27],x14                    \n\t" // Load c67  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x22                               \n\t"
"                                            \n\t"
" ld1 {v2.s}[0],[x27],x14                    \n\t" // Load c70  into quad and increment by rs_c.
" ld1 {v2.s}[1],[x27],x14                    \n\t" // Load c71  into quad and increment by rs_c.
" ld1 {v2.s}[2],[x27],x14                    \n\t" // Load c72  into quad and increment by rs_c.
" ld1 {v2.s}[3],[x27],x14                    \n\t" // Load c73  into quad and increment by rs_c.
" ld1 {v3.s}[0],[x27],x14                    \n\t" // Load c74  into quad and increment by rs_c.
" ld1 {v3.s}[1],[x27],x14                    \n\t" // Load c75  into quad and increment by rs_c.
" ld1 {v3.s}[2],[x27],x14                    \n\t" // Load c76  into quad and increment by rs_c.
" ld1 {v3.s}[3],[x27],x14                    \n\t" // Load c77  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x23                               \n\t"
"                                            \n\t"
" ld1 {v4.s}[0],[x27],x14                    \n\t" // Load c80  into quad and increment by rs_c.
" ld1 {v4.s}[1],[x27],x14                    \n\t" // Load c81  into quad and increment by rs_c.
" ld1 {v4.s}[2],[x27],x14                    \n\t" // Load c82  into quad and increment by rs_c.
" ld1 {v4.s}[3],[x27],x14                    \n\t" // Load c83  into quad and increment by rs_c.
" ld1 {v5.s}[0],[x27],x14                    \n\t" // Load c84  into quad and increment by rs_c.
" ld1 {v5.s}[1],[x27],x14                    \n\t" // Load c85  into quad and increment by rs_c.
" ld1 {v5.s}[2],[x27],x14                    \n\t" // Load c86  into quad and increment by rs_c.
" ld1 {v5.s}[3],[x27],x14                    \n\t" // Load c87  into quad and increment by rs_c.
"                                            \n\t"
" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS3:                     \n\t"
"                                            \n\t"
" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x21                               \n\t"
"                                            \n\t"
" st1 {v0.s}[0],[x27],x14                    \n\t" // Store c60  into quad and increment by rs_c.
" st1 {v0.s}[1],[x27],x14                    \n\t" // Store c61  into quad and increment by rs_c.
" st1 {v0.s}[2],[x27],x14                    \n\t" // Store c62  into quad and increment by rs_c.
" st1 {v0.s}[3],[x27],x14                    \n\t" // Store c63  into quad and increment by rs_c.
" st1 {v1.s}[0],[x27],x14                    \n\t" // Store c64  into quad and increment by rs_c.
" st1 {v1.s}[1],[x27],x14                    \n\t" // Store c65  into quad and increment by rs_c.
" st1 {v1.s}[2],[x27],x14                    \n\t" // Store c66  into quad and increment by rs_c.
" st1 {v1.s}[3],[x27],x14                    \n\t" // Store c67  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x22                               \n\t"
"                                            \n\t"
" st1 {v2.s}[0],[x27],x14                    \n\t" // Store c70  into quad and increment by rs_c.
" st1 {v2.s}[1],[x27],x14                    \n\t" // Store c71  into quad and increment by rs_c.
" st1 {v2.s}[2],[x27],x14                    \n\t" // Store c72  into quad and increment by rs_c.
" st1 {v2.s}[3],[x27],x14                    \n\t" // Store c73  into quad and increment by rs_c.
" st1 {v3.s}[0],[x27],x14                    \n\t" // Store c74  into quad and increment by rs_c.
" st1 {v3.s}[1],[x27],x14                    \n\t" // Store c75  into quad and increment by rs_c.
" st1 {v3.s}[2],[x27],x14                    \n\t" // Store c76  into quad and increment by rs_c.
" st1 {v3.s}[3],[x27],x14                    \n\t" // Store c77  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x23                               \n\t"
"                                            \n\t"
" st1 {v4.s}[0],[x27],x14                    \n\t" // Store c80  into quad and increment by rs_c.
" st1 {v4.s}[1],[x27],x14                    \n\t" // Store c81  into quad and increment by rs_c.
" st1 {v4.s}[2],[x27],x14                    \n\t" // Store c82  into quad and increment by rs_c.
" st1 {v4.s}[3],[x27],x14                    \n\t" // Store c83  into quad and increment by rs_c.
" st1 {v5.s}[0],[x27],x14                    \n\t" // Store c84  into quad and increment by rs_c.
" st1 {v5.s}[1],[x27],x14                    \n\t" // Store c85  into quad and increment by rs_c.
" st1 {v5.s}[2],[x27],x14                    \n\t" // Store c86  into quad and increment by rs_c.
" st1 {v5.s}[3],[x27],x14                    \n\t" // Store c87  into quad and increment by rs_c.
"                                            \n\t"
" dup  v8.4s, wzr                            \n\t"
" dup  v9.4s, wzr                            \n\t"
" dup  v10.4s, wzr                           \n\t"
" dup  v11.4s, wzr                           \n\t"
" dup  v12.4s, wzr                           \n\t"
" dup  v13.4s, wzr                           \n\t"
"                                            \n\t"
" fcmp s7,#0.0                               \n\t"
" beq .SBETAZEROGENSTOREDS4                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" mov x27, x24                               \n\t"
"                                            \n\t"
" ld1 {v8.s}[0],[x27],x14                    \n\t" // Load c90  into quad and increment by rs_c.
" ld1 {v8.s}[1],[x27],x14                    \n\t" // Load c91  into quad and increment by rs_c.
" ld1 {v8.s}[2],[x27],x14                    \n\t" // Load c92  into quad and increment by rs_c.
" ld1 {v8.s}[3],[x27],x14                    \n\t" // Load c93  into quad and increment by rs_c.
" ld1 {v9.s}[0],[x27],x14                    \n\t" // Load c94  into quad and increment by rs_c.
" ld1 {v9.s}[1],[x27],x14                    \n\t" // Load c95  into quad and increment by rs_c.
" ld1 {v9.s}[2],[x27],x14                    \n\t" // Load c96  into quad and increment by rs_c.
" ld1 {v9.s}[3],[x27],x14                    \n\t" // Load c97  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x25                               \n\t"
"                                            \n\t"
" ld1 {v10.s}[0],[x27],x14                   \n\t" // Load c100  into quad and increment by rs_c.
" ld1 {v10.s}[1],[x27],x14                   \n\t" // Load c101  into quad and increment by rs_c.
" ld1 {v10.s}[2],[x27],x14                   \n\t" // Load c102  into quad and increment by rs_c.
" ld1 {v10.s}[3],[x27],x14                   \n\t" // Load c103  into quad and increment by rs_c.
" ld1 {v11.s}[0],[x27],x14                   \n\t" // Load c104  into quad and increment by rs_c.
" ld1 {v11.s}[1],[x27],x14                   \n\t" // Load c105  into quad and increment by rs_c.
" ld1 {v11.s}[2],[x27],x14                   \n\t" // Load c106  into quad and increment by rs_c.
" ld1 {v11.s}[3],[x27],x14                   \n\t" // Load c107  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x26                               \n\t"
"                                            \n\t"
" ld1 {v12.s}[0],[x27],x14                   \n\t" // Load c110  into quad and increment by rs_c.
" ld1 {v12.s}[1],[x27],x14                   \n\t" // Load c111  into quad and increment by rs_c.
" ld1 {v12.s}[2],[x27],x14                   \n\t" // Load c112  into quad and increment by rs_c.
" ld1 {v12.s}[3],[x27],x14                   \n\t" // Load c113  into quad and increment by rs_c.
" ld1 {v13.s}[0],[x27],x14                   \n\t" // Load c114  into quad and increment by rs_c.
" ld1 {v13.s}[1],[x27],x14                   \n\t" // Load c115  into quad and increment by rs_c.
" ld1 {v13.s}[2],[x27],x14                   \n\t" // Load c116  into quad and increment by rs_c.
" ld1 {v13.s}[3],[x27],x14                   \n\t" // Load c117  into quad and increment by rs_c.
"                                            \n\t"
" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROGENSTOREDS4:                     \n\t"
"                                            \n\t"
" prfm pldl2keep,[x3]                        \n\t"
" prfm pldl2keep,[x4]                        \n\t"
"                                            \n\t"
" fmla v8.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v11.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v12.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
" fmla v13.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" mov x27, x24                               \n\t"
"                                            \n\t"
" st1 {v8.s}[0],[x27],x14                    \n\t" // Store c90  into quad and increment by rs_c.
" st1 {v8.s}[1],[x27],x14                    \n\t" // Store c91  into quad and increment by rs_c.
" st1 {v8.s}[2],[x27],x14                    \n\t" // Store c92  into quad and increment by rs_c.
" st1 {v8.s}[3],[x27],x14                    \n\t" // Store c93  into quad and increment by rs_c.
" st1 {v9.s}[0],[x27],x14                    \n\t" // Store c94  into quad and increment by rs_c.
" st1 {v9.s}[1],[x27],x14                    \n\t" // Store c95  into quad and increment by rs_c.
" st1 {v9.s}[2],[x27],x14                    \n\t" // Store c96  into quad and increment by rs_c.
" st1 {v9.s}[3],[x27],x14                    \n\t" // Store c97  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x25                               \n\t"
"                                            \n\t"
" st1 {v10.s}[0],[x27],x14                   \n\t" // Store c100  into quad and increment by rs_c.
" st1 {v10.s}[1],[x27],x14                   \n\t" // Store c101  into quad and increment by rs_c.
" st1 {v10.s}[2],[x27],x14                   \n\t" // Store c102  into quad and increment by rs_c.
" st1 {v10.s}[3],[x27],x14                   \n\t" // Store c103  into quad and increment by rs_c.
" st1 {v11.s}[0],[x27],x14                   \n\t" // Store c104  into quad and increment by rs_c.
" st1 {v11.s}[1],[x27],x14                   \n\t" // Store c105  into quad and increment by rs_c.
" st1 {v11.s}[2],[x27],x14                   \n\t" // Store c106  into quad and increment by rs_c.
" st1 {v11.s}[3],[x27],x14                   \n\t" // Store c107  into quad and increment by rs_c.
"                                            \n\t"
" mov x27, x26                               \n\t"
"                                            \n\t"
" st1 {v12.s}[0],[x27],x14                   \n\t" // Store c110  into quad and increment by rs_c.
" st1 {v12.s}[1],[x27],x14                   \n\t" // Store c111  into quad and increment by rs_c.
" st1 {v12.s}[2],[x27],x14                   \n\t" // Store c112  into quad and increment by rs_c.
" st1 {v12.s}[3],[x27],x14                   \n\t" // Store c113  into quad and increment by rs_c.
" st1 {v13.s}[0],[x27],x14                   \n\t" // Store c114  into quad and increment by rs_c.
" st1 {v13.s}[1],[x27],x14                   \n\t" // Store c115  into quad and increment by rs_c.
" st1 {v13.s}[2],[x27],x14                   \n\t" // Store c116  into quad and increment by rs_c.
" st1 {v13.s}[3],[x27],x14                   \n\t" // Store c147  into quad and increment by rs_c.*/
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
 "x24","x25","x26","x27",
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

void increasePrecissionV_HS
    (
       dim_t               n,
       _Float16*     restrict buffH,
       float*     restrict buffS
     )
{
    uint64_t i;
    
 /*   __asm__ volatile 
(
"                                            \n\t"
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
/*:// Register clobber list
 "x0", "x1", "x2","x3","x4",
 "x5", "x6", "x7", "x8",
 "x9", "x10","x11","x12",
 "x13","x14","x15",
 "x16","x17","x18","x19",       
 "x20","x21","x22","x23",
 "x24","x25","x26","x27",
 "v0", "v1", "v2", "v3",
 "v4", "v5", "v6", "v7",
 "v8", "v9", "v10","v11",
 "v12","v13","v14","v15",
 "v16","v17","v18","v19",
 "v20","v21","v22","v23",
 "v24","v25","v26","v27",
 "v28","v29","v30","v31"
);*/

    for(i = 0; i < n; i++)
        buffS[i] = buffH[i];

}

void decreasePrecissionV_SH(dim_t n, float*     restrict buffS, _Float16*     restrict buffH)
{
    uint64_t i;
    
    for(i = 0; i < n; i++)
        buffH[i] = buffS[i];
}
    

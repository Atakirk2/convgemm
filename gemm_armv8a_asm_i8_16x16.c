#include <blis.h>
//Unoptimized version developed to overcome hgemm_ref performance problmes
void hgemm_armv8a_asm_8x8
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
//" lsl x10,x9,#1                              \n\t" // cs_c * sizeof(int8) -- AUX.
"                                            \n\t" 
//" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
//" lsl x14,x13,#1                             \n\t" // rs_c * sizeof(int8).
"                                            \n\t"
" add x16,x2,x9                             \n\t" //Load address Column 1 of C
" add x17,x16,x9                            \n\t" //Load address Column 2 of C
" add x18,x17,x9                            \n\t" //Load address Column 3 of C
" add x19,x18,x9                            \n\t" //Load address Column 4 of C
" add x20,x19,x9                            \n\t" //Load address Column 5 of C
" add x21,x20,x9                            \n\t" //Load address Column 6 of C
" add x22,x21,x9                            \n\t" //Load address Column 7 of C
" add x23,x22,x9                            \n\t" //Load address Column 8 of C
" add x24,x23,x9                            \n\t" //Load address Column 9 of C
" add x25,x24,x9                            \n\t" //Load address Column 10 of C
" add x26,x25,x9                            \n\t" //Load address Column 11 of C
" add x27,x26,x9                            \n\t" //Load address Column 12 of C
" add x28,x27,x9                            \n\t" //Load address Column 13 of C
" add x29,x28,x9                            \n\t" //Load address Column 14 of C
" add x30,x29,x9                            \n\t" //Load address Column 15 of C



"                                            \n\t" //Loads can cause unauthorized acces if k < 2
" ld1 {v0.16b, v1.16b}, [x0]                   \n\t" //Load a 
"                                            \n\t"
" ld1 {v4.16b, v5.16b}, [x1]                   \n\t" //Load b
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
" prfm pldl1keep,[x29]                       \n\t" // Prefetch c.
" prfm pldl1keep,[x30]                       \n\t" // Prefetch c.




"                                            \n\t"
" dup  v8.16b, wzr                            \n\t" // Vectors for accummulating column 0
" dup  v9.16b, wzr                            \n\t" 
" dup  v10.16b, wzr                           \n\t"
" dup  v11.16b, wzr                           \n\t" // Vector for accummulating column 1
" dup  v12.16b, wzr                           \n\t" 
" dup  v13.16b, wzr                           \n\t" 
" dup  v14.16b, wzr                           \n\t" // Vector for accummulating column 2
//" prfm    PLDL1KEEP, [x0, #64]              \n\t"
" dup  v15.16b, wzr                           \n\t" 
//" prfm    PLDL1KEEP, [x0, #128]              \n\t"
" dup  v16.16b, wzr                           \n\t" 
" dup  v17.16b, wzr                           \n\t" 
" dup  v18.16b, wzr                           \n\t" 
" dup  v19.16b, wzr                           \n\t" 
" dup  v20.16b, wzr                           \n\t" 
" dup  v21.16b, wzr                           \n\t" 
" dup  v22.16b, wzr                           \n\t" 
" dup  v23.16b, wzr                           \n\t" 

"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .SCONSIDERKLEFT                        \n\t"
"                                            \n\t"
"add x0, x0, #32                             \n\t" //update address of A
"add x1, x1, #32                             \n\t" //update address of B
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .SLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" .SLOOPKITER:                               \n\t" // Body of the k_iter loop.
"                                            \n\t"

" mla v8.16b, v0.16b,v4.b[0]                  \n\t" // Accummulate.
" mla v9.16b,v0.16b,v4.b[1]                   \n\t" // Accummulate.
" ld1 {v2.16b, v3.16b}, [x0],#32               \n\t"
" mla v10.16b,v0.16b,v4.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v0.16b,v4.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v0.16b,v4.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v0.16b,v4.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v0.16b,v4.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v0.16b,v4.b[7]                  \n\t" // Accummulate.
" ld1 {v6.16b, v7.16b}, [x1],#32               \n\t"
" mla v16.16b,v0.16b,v4.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v0.16b,v4.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v0.16b,v4.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v0.16b,v4.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v0.16b,v4.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v0.16b,v4.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v0.16b,v4.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v0.16b,v4.b[15]                  \n\t" // Accummulate.
"                                            \n\t"

//" prfm    PLDL1KEEP, [x1, #32]              \n\t" 
"                                            \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
"                                            \n\t"
//" prfm    PLDL1KEEP, [x0, #48]              \n\t"
//" prfm    PLDL1KEEP, [x0, #112]              \n\t"
//" prfm    PLDL1KEEP, [x0, #176]              \n\t"
"                                            \n\t"
" mla v8.16b, v1.16b,v5.b[0]                  \n\t" // Accummulate.
" mla v9.16b, v1.16b,v5.b[1]                   \n\t" // Accummulate.
" mla v10.16b,v1.16b,v5.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v1.16b,v5.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v1.16b,v5.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v1.16b,v5.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v1.16b,v5.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v1.16b,v5.b[7]                  \n\t" // Accummulate.
" mla v16.16b,v1.16b,v5.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v1.16b,v5.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v1.16b,v5.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v1.16b,v5.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v1.16b,v5.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v1.16b,v5.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v1.16b,v5.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v1.16b,v5.b[15]                  \n\t" // Accummulate.
"                                            \n\t" //End It 2
"                                            \n\t"
" mla v8.16b, v2.16b,v6.b[0]                  \n\t" // Accummulate.
" mla v9.16b, v2.16b,v6.b[1]                   \n\t" // Accummulate.
" ld1 {v0.16b, v1.16b}, [x0],#32               \n\t"
" mla v10.16b,v2.16b,v6.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v2.16b,v6.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v2.16b,v6.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v2.16b,v6.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v2.16b,v6.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v2.16b,v6.b[7]                  \n\t" // Accummulate.
" ld1 {v4.16b, v5.16b}, [x1],#32               \n\t"
" mla v15.16b,v2.16b,v6.b[8]                  \n\t" // Accummulate.
" mla v16.16b,v2.16b,v6.b[9]                  \n\t" // Accummulate.
" mla v17.16b,v2.16b,v6.b[10]                  \n\t" // Accummulate.
" mla v18.16b,v2.16b,v6.b[11]                  \n\t" // Accummulate.
" mla v19.16b,v2.16b,v6.b[12]                  \n\t" // Accummulate.
" mla v20.16b,v2.16b,v6.b[13]                  \n\t" // Accummulate.
" mla v21.16b,v2.16b,v6.b[14]                  \n\t" // Accummulate.
" mla v2.16b,v2.16b,v6.b[15]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" mla v8.16b, v3.16b,v7.b[0]                  \n\t" // Accummulate.
" mla v9.16b, v3.16b,v7.b[1]                   \n\t" // Accummulate.
" mla v10.16b,v3.16b,v7.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v3.16b,v7.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v3.16b,v7.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v3.16b,v7.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v3.16b,v7.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v3.16b,v7.b[7]                  \n\t" // Accummulate.
" mla v16.16b,v3.16b,v7.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v3.16b,v7.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v3.16b,v7.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v3.16b,v7.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v3.16b,v7.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v3.16b,v7.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v3.16b,v7.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v3.16b,v7.b[15]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 4
" sub x5,x5,1                                \n\t" // i-=1.
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .SLOOPKITER                            \n\t"
"                                            \n\t" 
" .SLASTITER:                                \n\t" // Last iteration of k_iter loop.
"                                            \n\t" 
" mla v8.16b, v0.16b,v4.b[0]                  \n\t" // Accummulate.
" mla v9.16b,v0.16b,v4.b[1]                   \n\t" // Accummulate.
" ld1 {v2.16b, v3.16b}, [x0],#32               \n\t"
" mla v10.16b,v0.16b,v4.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v0.16b,v4.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v0.16b,v4.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v0.16b,v4.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v0.16b,v4.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v0.16b,v4.b[7]                  \n\t" // Accummulate.
" ld1 {v6.16b, v7.16b}, [x1],#32               \n\t"
" mla v16.16b,v0.16b,v4.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v0.16b,v4.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v0.16b,v4.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v0.16b,v4.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v0.16b,v4.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v0.16b,v4.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v0.16b,v4.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v0.16b,v4.b[15]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 1
"                                            \n\t"
" mla v8.16b, v1.16b,v5.b[0]                  \n\t" // Accummulate.
" mla v9.16b, v1.16b,v5.b[1]                   \n\t" // Accummulate.
" mla v10.16b,v1.16b,v5.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v1.16b,v5.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v1.16b,v5.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v1.16b,v5.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v1.16b,v5.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v1.16b,v5.b[7]                  \n\t" // Accummulate.
" mla v16.16b,v1.16b,v5.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v1.16b,v5.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v1.16b,v5.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v1.16b,v5.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v1.16b,v5.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v1.16b,v5.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v1.16b,v5.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v1.16b,v5.b[15]                  \n\t" // Accummulate.
"                                            \n\t" //End It 2
"                                            \n\t"
" mla v8.16b, v2.16b,v6.b[0]                  \n\t" // Accummulate.
" mla v9.16b, v2.16b,v6.b[1]                   \n\t" // Accummulate.
" mla v10.16b,v2.16b,v6.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v2.16b,v6.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v2.16b,v6.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v2.16b,v6.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v2.16b,v6.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v2.16b,v6.b[7]                  \n\t" // Accummulate.
" mla v15.16b,v2.16b,v6.b[8]                  \n\t" // Accummulate.
" mla v16.16b,v2.16b,v6.b[9]                  \n\t" // Accummulate.
" mla v17.16b,v2.16b,v6.b[10]                  \n\t" // Accummulate.
" mla v18.16b,v2.16b,v6.b[11]                  \n\t" // Accummulate.
" mla v19.16b,v2.16b,v6.b[12]                  \n\t" // Accummulate.
" mla v20.16b,v2.16b,v6.b[13]                  \n\t" // Accummulate.
" mla v21.16b,v2.16b,v6.b[14]                  \n\t" // Accummulate.
" mla v2.16b,v2.16b,v6.b[15]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t" //End It 3
"                                            \n\t"
" mla v8.16b, v3.16b,v7.b[0]                  \n\t" // Accummulate.
" mla v9.16b, v3.16b,v7.b[1]                   \n\t" // Accummulate.
" mla v10.16b,v3.16b,v7.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v3.16b,v7.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v3.16b,v7.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v3.16b,v7.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v3.16b,v7.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v3.16b,v7.b[7]                  \n\t" // Accummulate.
" mla v16.16b,v3.16b,v7.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v3.16b,v7.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v3.16b,v7.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v3.16b,v7.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v3.16b,v7.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v3.16b,v7.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v3.16b,v7.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v3.16b,v7.b[15]                  \n\t" // Accummulate.
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
" ld1 {v0.16b}, [x0],#16                      \n\t"
" ld1 {v4.16b}, [x1],#16        \n\t"
" mla v8.16b, v0.16b,v4.b[0]                  \n\t" // Accummulate.
" mla v9.16b,v0.16b,v4.b[1]                  \n\t" // Accummulate.
" mla v10.16b,v0.16b,v4.b[2]                  \n\t" // Accummulate.
" mla v11.16b,v0.16b,v4.b[3]                  \n\t" // Accummulate.
" mla v12.16b,v0.16b,v4.b[4]                  \n\t" // Accummulate.
" mla v13.16b,v0.16b,v4.b[5]                  \n\t" // Accummulate.
" mla v14.16b,v0.16b,v4.b[6]                  \n\t" // Accummulate.
" mla v15.16b,v0.16b,v4.b[7]                  \n\t" // Accummulate.
" mla v16.16b,v0.16b,v4.b[8]                  \n\t" // Accummulate.
" mla v17.16b,v0.16b,v4.b[9]                  \n\t" // Accummulate.
" mla v18.16b,v0.16b,v4.b[10]                  \n\t" // Accummulate.
" mla v19.16b,v0.16b,v4.b[11]                  \n\t" // Accummulate.
" mla v20.16b,v0.16b,v4.b[12]                  \n\t" // Accummulate.
" mla v21.16b,v0.16b,v4.b[13]                  \n\t" // Accummulate.
" mla v22.16b,v0.16b,v4.b[14]                  \n\t" // Accummulate.
" mla v23.16b,v0.16b,v4.b[15]                  \n\t" // Accummulate.
"                                            \n\t"
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .SLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" .SPOSTACCUM:                               \n\t"
"                                            \n\t"
//" ld1r {v6.16b},[x7]                          \n\t" // Load alpha.
//" ld1r {v7.16b},[x8]                          \n\t" // Load beta
" ld1 {v1.h}[0],[x7]                          \n\t" // Load alpha.
" ld1 {v5.h}[0],[x8]                          \n\t" // Load beta
"                                            \n\t"
//" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
//" bne .SGENSTORED                            \n\t"
"                                            \n\t"
" .SCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
"                                            \n\t"
" dup  v24.16b, wzr                            \n\t"
" dup  v25.16b, wzr                            \n\t"
" dup  v26.16b, wzr                            \n\t"
" dup  v27.16b, wzr                            \n\t"
" dup  v28.16b, wzr                            \n\t"
" dup  v29.16b, wzr                            \n\t"
" dup  v30.16b, wzr                            \n\t"
" dup  v31.16b, wzr                            \n\t"
"                                            \n\t"
" fcmp h5,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS1                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v24.16b}, [x2]                         \n\t" //Load column 0 of C
" ld1 {v25.16b}, [x16]                        \n\t" //Load column 1 of C
" ld1 {v26.16b}, [x17]                         \n\t" //Load column 2 of C
" ld1 {v27.16b}, [x18]                        \n\t" //Load column 3 of C
" ld1 {v28.16b}, [x19]                         \n\t" //Load column 4 of C
" ld1 {v29.16b}, [x20]                        \n\t" //Load column 5 of C
" ld1 {v30.16b}, [x21]                         \n\t" //Load column 6 of C
" ld1 {v31.16b}, [x22]                        \n\t" //Load column 7 of C
"                                            \n\t"
" mul v24.16b,v24.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v25.16b,v25.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v26.16b,v26.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v27.16b,v27.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v28.16b,v28.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v29.16b,v29.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v30.16b,v30.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v31.16b,v31.16b,v5.b[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS1:                     \n\t"
"                                            \n\t"
" mla v24.16b,v8.16b,v1.b[0]                   \n\t" // Scale by alpha
" mla v25.16b,v9.16b,v1.b[0]                   \n\t" // Scale by alpha
" mla v26.16b,v10.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v27.16b,v11.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v28.16b,v12.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v29.16b,v13.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v30.16b,v14.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v31.16b,v15.16b,v1.b[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v24.16b}, [x2]                         \n\t" //Store column 0 of C
" st1 {v25.16b}, [x16]                        \n\t" //Store column 1 of C
" st1 {v26.16b}, [x17]                         \n\t" //Store column 2 of C
" st1 {v27.16b}, [x18]                        \n\t" //Store column 3 of C
" st1 {v28.16b}, [x19]                         \n\t" //Store column 4 of C
" st1 {v29.16b}, [x20]                        \n\t" //Store column 5 of C
" st1 {v30.16b}, [x21]                         \n\t" //Store column 6 of C
" st1 {v31.16b}, [x22]                        \n\t" //Store column 7 of C
"                                            \n\t"
"                                            \n\t"
" dup  v6.16b, wzr                            \n\t"
" dup  v7.16b, wzr                            \n\t"
" dup  v8.16b, wzr                            \n\t"
" dup  v9.16b, wzr                            \n\t"
" dup  v10.16b, wzr                            \n\t"
" dup  v11.16b, wzr                            \n\t"
" dup  v12.16b, wzr                            \n\t"
" dup  v13.16b, wzr                            \n\t"
"                                            \n\t"
" fcmp h5,#0.0                               \n\t"
" beq .SBETAZEROCOLSTOREDS2                  \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
" ld1 {v6.16b}, [x23]                         \n\t" //Load column 8 of C
" ld1 {v7.16b}, [x24]                        \n\t" //Load column 9 of C
" ld1 {v8.16b}, [x25]                         \n\t" //Load column 10 of C
" ld1 {v9.16b}, [x26]                        \n\t" //Load column 11 of C
" ld1 {v10.16b}, [x27]                         \n\t" //Load column 12 of C
" ld1 {v11.16b}, [x28]                        \n\t" //Load column 13 of C
" ld1 {v12.16b}, [x29]                         \n\t" //Load column 14 of C
" ld1 {v13.16b}, [x30]                        \n\t" //Load column 15 of C
"                                            \n\t"
" mul v6.16b,v6.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v7.16b,v7.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v8.16b,v8.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v9.16b,v9.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v10.16b,v10.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v11.16b,v11.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v12.16b,v12.16b,v5.b[0]                   \n\t" // Scale by beta
" mul v13.16b,v13.16b,v5.b[0]                   \n\t" // Scale by beta
"                                            \n\t"
" .SBETAZEROCOLSTOREDS2:                     \n\t"
"                                            \n\t"
" mla v6.16b,v16.16b,v1.b[0]                   \n\t" // Scale by alpha
" mla v7.16b,v17.16b,v1.b[0]                   \n\t" // Scale by alpha
" mla v8.16b,v18.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v9.16b,v19.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v10.16b,v20.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v11.16b,v21.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v12.16b,v22.16b,v1.b[0]                  \n\t" // Scale by alpha
" mla v13.16b,v23.16b,v1.b[0]                  \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v6.16b}, [x23]                         \n\t" //Store column 0 of C
" st1 {v7.16b}, [x24]                        \n\t" //Store column 1 of C
" st1 {v8.16b}, [x25]                         \n\t" //Store column 2 of C
" st1 {v9.16b}, [x26]                        \n\t" //Store column 3 of C
" st1 {v10.16b}, [x27]                         \n\t" //Store column 4 of C
" st1 {v11.16b}, [x28]                        \n\t" //Store column 5 of C
" st1 {v12.16b}, [x29]                         \n\t" //Store column 6 of C
" st1 {v13.16b}, [x30]                        \n\t" //Store column 7 of C
"                                            \n\t"
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
 "x20","x21","x22", "x23",
 "x24","x25","x26", "x27",
 "x28","x29","x30",
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

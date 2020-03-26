INCLUDE=/home/jetsonuser/libs/include/blis/
CC=gcc
CFLAGS= -Wl,-rpath,/home/jetsonuser/libs/lib/ $(COPTFLAGS)
COPTFLAGS= -O3 -ftree-vectorize -mtune=cortex-a57 -march=armv8-a+fp+simd -mcpu=cortex-a57 -funsafe-math-optimizations -ffp-contract=fast -fopt-info-vec-optimized=vecOpt.out
LIB= -lblis -lm
uKOBJS= gemm_ref.o gemm_armv8a_asm_d6x8.o

.PHONY: all clean test comp

all: testIm2Col.x compGEMM.x convEval.x
test: testIm2Col.x
comp: compGEMM.x
eval: convEval.x

%.o: %.c 
	$(CC) -c -o $@ $< $(CFLAGS) -I$(INCLUDE)
	
gemmConv.o: gemmConv.c $(uKOBJS) gemmConv.h
	$(CC) -c -o $@ $< $(CFLAGS) -I$(INCLUDE)
	
convCommon.o: convCommon.c convCommon.h
	$(CC) -c -o $@ $< $(CFLAGS) -I$(INCLUDE)

%.x: %.o gemmConv.o convCommon.o 
	$(CC) -o $@ $^ $(uKOBJS) $(CFLAGS) $(LIB)

clean:
	rm *.o
	rm vecOpt.out 

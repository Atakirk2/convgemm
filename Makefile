include paths.mk

CC=gcc-10
CFLAGS= -Wl,-rpath,$(LIBPATH) -L$(LIBPATH) -fpic -fopenmp $(COPTFLAGS) $(OPTS)
ARCH=A57
ifeq ($(ARCH),Carmel)
	ARCHFLAGS= -mtune=cortex-a76  -march=armv8.2-a+fp16fml -Dfp16_support #NVIDIA Carmel
else ifeq ($(ARCH),A57)
	ARCHFLAGS= -mtune=cortex-a57 -march=armv8-a+fp+simd -mcpu=cortex-a57 #Cortex A-57
else ifeq ($(ARCH),A72) # Raspberry Pi 4
	ARCHFLAGS= -mtune=cortex-a72 -march=armv8-a+fp+crc -mcpu=cortex-a72 
endif
COPTFLAGS= -O3 -ftree-vectorize $(ARCHFLAGS) -funsafe-math-optimizations -ffp-contract=fast 
LIB= -lblis -lm -lgemmConv 
ifneq (,$(findstring -DPWR, $(OPTS)))
	LIB+= -lpmlib
endif
uKOBJS= gemm_ref.o gemm_armv8a_asm_d6x8.o  gemm_armv8a_neon_s8x12.o gemm_armv8a_asm_s8x12_v2.o gemm_armv8a_asm_hs8x12.o  gemm_armv8a_asm_i16_24x8.o

ifeq ($(ARCH),Carmel)
uKOBJS+=gemm_armv8a_asm_h8x24.o gemm_armv8a_asm_h24x8.o gemm_armv8a_asm_h8x8.o
endif

.PHONY: all clean test comp

all: testIm2Col.x compGEMM.x convEval.x
test: testIm2Col.x
testCol: testCol2Im.x
comp: compGEMM.x
eval: convEval.x
micro: testMicrokernels.x
blocks: evalBlockSize.x
peakPerf: peakPerfTest.x
lib: $(LIBPATH)/libgemmConv.so

%.o: %.c 
	$(CC) -c -o $@ $< $(CFLAGS) -I$(INCLUDE)
	
convGemm.o: convGemm.c $(uKOBJS) convGemm.h
	$(CC) -c -o $@ $< $(CFLAGS) -I$(INCLUDE)

$(LIBPATH)/libgemmConv.so: convGemm.o 
	$(CC) -shared -fopenmp -o $@  $< $(uKOBJS)
	
convCommon.o: convCommon.c convCommon.h
	$(CC) -c -o $@ $< $(CFLAGS) -I$(INCLUDE)

peakPerfTest.x: peakPerfTest.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

testMicrokernels.x: testMicrokernels.o convGemm.o
	$(CC) -o $@ $^ $(uKOBJS) $(CFLAGS) $(LIB)	


%.x: %.o $(LIBPATH)/libgemmConv.so convCommon.o 
	$(CC) -o $@ $^  $(CFLAGS) $(LIB)

clean:
	rm *.o


Introduction
-------------
This reposirory contains the implementation of several programs centered around the convolution operator for CNNS.

Compilation
------------
This repository contains 4 different programs, they can be all compiled with:

`make` or `make all`

Each program has its own target and some compilation options taht should be pased to make as:

`make OPTS="Options"`

- `make comp`:  compiles the gemm comparator. This program compares the gemm implemented in BLIS with the custom GEMM (based in BLIS architecture) used in this project. Options:
    -Datatype options, exclusive, default= single precision(FP32):
        - `-Dfp_D`: Use double precision(FP64) as datatype.
        - `-Dfp_H`: Use half precision(FP16)  as datatype.
        - `-Dfp_HS`: Use half precision storage with single precision arithmetics.
        - `-Di_16`: Use 16 bits integers  as datatype
    - `-DCOMPARE` : Compare the result of both GEMM and emit an error measure.
    - `-DPWR` : Perform power and energy measurements together with the performance measutrements. Needs running PMLIB server on loopback interface.
    

- `make test`: compiles the convolution tester. This program compares a naive convolution algorithm against the im2col + gemm approach and against our CONVGEMM algorithm (convolution using a gemm with implicit im2col).

- `make eval` :  compiles the CNN evaluator. This program evaluates the different implementations of the convolution operator for different CNNS. Options:
    - `-DNONAIVE` : Skips the execution of the naive convolution to save time.
    - `-DLAYER_EVAL` : Performs the evaluation layer per layer and then agregates the results.
    - `-Deval_precision`: Performs a precision comparison evaluating the model using the CONVGEMM algorithm with FP16 and FP32.
    - `-Dout_csv` : Formats the output as a semicolon separated CSV.

- `make blocks`: compiles a block size evaluator to obtain experimentaly the optimal K_c and M_c values for a given architecture. 

The compilation is optimized for ARM Cortex A-57 and NVIDIA Carmel.

Referencing
---------------

P. San Juan, A. Castelló, M. F. Dolz, P. Alonso-Jordá and E. S. Quintana-Ortí, "High Performance and Portable Convolution Operators for Multicore Processors," 2020 IEEE 32nd International Symposium on Computer Architecture and High Performance Computing (SBAC-PAD), Porto, Portugal, 2020, pp. 91-98, doi: 10.1109/SBAC-PAD49847.2020.00023.


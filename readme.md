Introduction
-------------
This reposirory contains the implementation of several programs centered around the convolution operator for CNNS.

Compilation
------------
This repository contains 3 different programs, they can be all compiled with:

`make` or `make all`

but each one has its own target too:

- `make comp`  compiles the gemm comparator. This program compares the gemm implemented in BLIS with the custom GEMM (based in BLIS architecture) used in this project.

- `make test` compiles the convolution tester. This program compares a naive convolution algorithm against the im2col + gemm approach and agains our CONVGEMM algorithm (convolution using a gemm with implicit im2col).


- `make eval` :  compiles the CNN evaluator. This program evaluates the different implementations of the convolution operator for different CNNS.


The compilation is optimized for ARM Cortex A-57.

Referencing
---------------

#! /bin/bash -l

set -e

module load CUDA/10.1.105
module load foss/2018a

# clean up
rm -f *.o *.exe

# build extra libraries
#gcc -std=gnu99 -c ../../extras/compare/compare_arrays.c -o ../../extras/compare/compare_arrays.o
gcc -std=gnu99 -c ../../extras/initialize/init.c -o ../../extras/initialize/init.o
gcc -std=gnu99 -c ../../extras/timer/timer.c -o ../../extras/timer/timer.o

# build cuda kernels
src1=3.3-reduce-float
nvcc -arch=sm_35 -I../../extras/initialize -I../../extras/preprocessor -I../../extras/timer -c ${src1}.cu -o ${src1}.o 
nvcc -arch=sm_35 ${src1}.o ../../extras/initialize/init.o ../../extras/timer/timer.o -o ${src1}.exe  

src2=3.4-dynamic-parallelism
nvcc -arch=sm_35 -rdc=true -lcudadevrt ${src2}.cu -o ${src2}.exe


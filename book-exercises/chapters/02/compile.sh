#! /bin/bash -l

set -e

module load CUDA/10.1.105
module load foss/2018a

# clean up
rm -f *.o *.exe

# build extra libraries
gcc -std=gnu99 -c ../../extras/compare/compare_arrays.c -o ../../extras/compare/compare_arrays.o

# build cuda kernels
cu_files=$(ls *.cu)
for src in $cu_files; do
  stem=${src::-3}
  echo compiling ... ${stem}.exe
  nvcc -arch=sm_30 -O3 -g -c $src -I../../extras/compare -o ${stem}.o

  # link libraries with kernels
  nvcc -o ${stem}.exe ${stem}.o ../../extras/compare/compare_arrays.o
done

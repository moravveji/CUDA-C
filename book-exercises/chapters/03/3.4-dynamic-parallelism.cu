#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(cmnd) {     \
  cudaError_t ierr = cmnd;   \
  if (ierr != cudaSuccess) { \
    printf("Error: CHECK: %s(%d): %s\n", __LINE__, __FILE__, cudaGetErrorString(ierr)); \
    return ierr; \
  }              \
}

__global__ void nestedHellowWorld(int const size, int depth) {
  int thx = threadIdx.x;
  printf("Hello World: depth=%d, threadIdx.x=%d, blockIdx.x=%d \n", \
          depth, thx, blockIdx.x);

  if (size == 1) return; // return if size is one

  int nthreads = size >> 1; // reduce blockDim by half

  if (thx == 0 && nthreads > 0) {
    nestedHellowWorld <<< 1, nthreads >>> (nthreads, ++depth); 
    printf("################### Nested execution depth = %d \n", depth);
  }
}

int main(int argc, char ** argv) {
  const int dev = 0;
  CHECK(cudaSetDevice(dev)); 
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Info: Usisng device: %s\n", prop.name);

  const int tpb = 16;
  const int grid = 1;

  nestedHellowWorld <<< grid, tpb >>> (tpb, 0);
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaDeviceReset());
  return 0;
}
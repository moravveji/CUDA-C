#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void) {
  printf("Hello from GPU thread: %d!\n", threadIdx.x);
}

int main(void){
  // CPU says hi
  printf("Hello from CPU! \n");

  // GPU says hi
  helloFromGPU<<<1, 4>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  
  return 0;
}

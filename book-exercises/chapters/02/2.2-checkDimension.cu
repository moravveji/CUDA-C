#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
  printf("threadIdx: (%d, %d, %d) \n"
         "blockIdx:  (%d, %d, %d) \n"
         "blockDim:  (%d, %d, %d) \n"
         "gridDim:   (%d, %d, %d) \n", 
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z, 
          blockDim.x, blockDim.y, blockDim.z,
          gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv){
  const int nElem = 12;
  dim3 block(4);
  dim3 grid( (nElem + block.x - 1)/block.x );

  // check the grid and block sizes on the host
  printf("Block: (%d, %d, %d) \n"
         "Grid:  (%d, %d, %d) \n", 
          block.x, block.y, block.z, grid.x, grid.y, grid.z);

  // check the grid and block sizes on the device
  checkIndex <<< grid, block >>>();

  // done
  cudaDeviceReset();

  return 0;
}

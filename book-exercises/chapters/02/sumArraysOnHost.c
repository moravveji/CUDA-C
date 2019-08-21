#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float const *a, float const *b, float *c, const int n) {
  for (int k=0; k<n; k++) {c[k] = a[k] + b[k];}
}

__global__ sumArrayOnDevice(float const *a, float const *b, float *c) {
  int k = threadIdx.x
  c[k] = a[k] + b[k];
}

void init_rand(float *arr, const int n) {
  time_t t;
  srand( (unsigned int) time(&t) );

  for (int k=0; k<n; k++) { arr[k] = (float)( rand() & 0xFF )/10.0f; }
}

int main(int argc, char **argv) {

  // CPU code: 
  // declare vars
  const int nElem = 1024;
  const int nBytes = nElem * sizeof(float);
  float *h_a, *h_b, *h_c;
  h_a = (float *)malloc(nBytes);
  h_b = (float *)malloc(nBytes);
  h_c = (float *)malloc(nBytes);

  // initialize with random values
  init_rand(h_a, nElem);
  init_rand(h_b, nElem);

  // add two arrays element-by-element
  sumArraysOnHost(h_a, h_b, h_c, nElem);

  // GPU code
  const int tbd = 64;
  const int nblocks = (nBytes + tbd - 1) / tbd;
  dim3 block(tbd);
  dim3 grid(nblocks);
  float * gpuRef = (float *) malloc(nBytes);
  float *d_a, *d_b, *d_c;
  cudaMalloc((float **)&d_a, nBytes);
  cudaMalloc((float **)&d_b, nBytes);
  cudaMalloc((float **)&d_c, nBytes);

  printf("grid.x %d grid.y %d grid.z %d\n",grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n",block.x, block.y, block.z);

  // copy data from host to device
  cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

  // kernel launch
  sumArrayOnDevice<<<grid, block>>> (*d_a, *d_b, *d_c);
  cudaError_t cudaDeviceSynchronize();

  // copy results back to the host
  cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);

  // difference between the host and device calculation
  float mean_diff = 0.0f;
  for (int k=0; k<nElem; k++) { mean_diff += (gpuRef[k] - h_c[k]); }
  mean_diff /= nElem;
  printf("\nMean difference between device and host = %16.8f\n\n", mean_diff);

  // free the device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
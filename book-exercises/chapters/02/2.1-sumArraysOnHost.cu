#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// Error handling
#define check(cmnd)                                                                        \
{                                                                                          \
  const cudaError_t error = cmnd;                                                          \
  if (error != cudaSuccess) {                                                              \
    printf("Error %d: %s:%d. %s\n", error, __FILE__, __LINE__, cudaGetErrorString(error)); \
    exit(error);                                                                           \
  }                                                                                        \
}                                                                                          

void checkResults(float * arr1, float * arr2, int n) {
  float mean_diff = 0.0f;
  bool match = 1;
  double epsilon = 1.0e-8;

  for (int k=0; k<n; k++) { 
    mean_diff += (arr1[k] - arr2[k]); 
    if (abs(arr1[k] - arr2[k]) > epsilon) {
      match = 0;
      printf("Error: checkResults: arr1[%d]=%16.8f but arr2[%d]=%16.8f\n", k, arr1[k], k, arr2[k]);
      break; 
    }
  }
  mean_diff = mean_diff / n;
  printf("%s\n", "loop finished");
  
  printf("\nMean difference between device and host = %16.8f\n\n", mean_diff);
  // printf("arr1[%d]=%16.8f; arr2[%d]=%16.8f\n", n-1, arr1[n-1], n-1, arr2[n-1]);  
  if (match) { printf("Info: Arrays match!\n"); }
}

// Function definitions
void sumArraysOnHost(float const *a, float const *b, float *c, const int n) {
  for (int k=0; k<n; k++) {c[k] = a[k] + b[k];}
}

// device function
__global__ void sumArrayOnDevice(float const *a, float const *b, float *c, const int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) { c[k] = a[k] + b[k]; }
}

// host random init 
void init_rand(float *arr, const int n) {
  time_t t;
  srand( (unsigned int) time(&t) );

  for (int k=0; k<n; k++) { arr[k] = (float)( rand() & 0xFF )/10.0f; }
}

// the main function
int main(int argc, char **argv) {
  printf("%s starting ... \n", argv[0]);

  // CPU code: 
  // declare vars
  const int nElem = 1 << 10;
  const size_t nBytes = nElem * sizeof(float);
  float *h_a, *h_b, *hostRef, *gpuRef;
  h_a     = (float *)malloc(nBytes);
  h_b     = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef  = (float *)malloc(nBytes);

  // initialize with random values
  memset(hostRef, 0, nBytes);
  memset(gpuRef,  0, nBytes);
  init_rand(h_a, nElem);
  init_rand(h_b, nElem);

  // add two arrays element-by-element
  sumArraysOnHost(h_a, h_b, hostRef, nElem);

  // GPU code
  int const device = 0;
  check(cudaSetDevice(device));

  const int tbd = 64;
  const int nblocks = (nBytes + tbd - 1) / tbd;
  dim3 block(tbd);
  dim3 grid(nblocks);
  float *d_a, *d_b, *d_c;
  check(cudaMalloc((float **)&d_a, nBytes));
  check(cudaMalloc((float **)&d_b, nBytes));
  check(cudaMalloc((float **)&d_c, nBytes));

  printf("grid.x %d grid.y %d grid.z %d\n",grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n",block.x, block.y, block.z);

  // copy data from host to device
  check(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));

  // kernel launch
  sumArrayOnDevice<<<grid, block>>> (d_a, d_b, d_c, nElem);
  check(cudaDeviceSynchronize());

  // copy results back to the host
  check(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));

  // difference between the host and device calculation
  checkResults(gpuRef, hostRef, nElem);

  // free the host & device memory
  free(h_a); free(h_b); free(hostRef); free(gpuRef);
  check(cudaFree(d_a));
  check(cudaFree(d_b));
  check(cudaFree(d_c));

  return 0;
}

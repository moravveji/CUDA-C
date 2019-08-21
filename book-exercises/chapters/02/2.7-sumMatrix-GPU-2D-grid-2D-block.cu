#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
extern "C" { 
  #include "compare_arrays.h"
}

#define CHECK(cmnd) {                                                        \
  cudaError_t err = cmnd;                                                    \
  if (err != cudaSuccess) {                                                  \
    printf("Error: %s(%d): %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(err);                                                               \
  }                                                                          \
}

void sumMatrixOnHost(float * a, float * b, float * c, const size_t nx, const size_t ny) {
  float * ma = a;
  float * mb = b;
  float * mc = c;
  for (size_t iy=0; iy<ny; iy++) {
    for (size_t ix=0; ix<nx; ix++) {
      mc[ix] = ma[ix] + mb[ix];
    }
    ma += nx; mb += nx; mc += nx;
  }
}

__global__ void sumMatrixOnGPU(float const * ma, float const * mb, float * mc, const size_t nx, const size_t ny) {
  size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  size_t iy = threadIdx.y + blockIdx.y * blockDim.y;
  size_t id = iy * nx + ix;
  if (ix < nx && iy < ny) mc[id] = ma[id] + mb[id];
}

void initMtrx(float *mtrx, const int nxy) {
  time_t t;
  srand((unsigned int) time(&t));
  float * ptr = mtrx;
  for (int k=0; k<nxy; k++) {
    ptr[k] = ( (float) (rand() & 0xFF))/10.0;
  }
}

int main(int argc, char ** argv) {

  // problem size
  const size_t nx = 1 << 14;
  const size_t ny = 1 << 14;
  const size_t nxy= nx * ny;
  const size_t n_bytes = nxy * sizeof(float); 

  // host array
  float *h_ma, *h_mb, *h_ref, *h_gpu;
  h_ma = (float *)malloc(n_bytes);
  h_mb = (float *)malloc(n_bytes);
  h_ref= (float *)malloc(n_bytes);
  h_gpu= (float *)malloc(n_bytes);
  initMtrx(h_ma, nxy);
  initMtrx(h_mb, nxy);
  memset(h_ref, 0.0, n_bytes);
  memset(h_gpu, 0.0, n_bytes);

  sumMatrixOnHost(h_ma, h_mb, h_ref, nx, ny);

  // GPU kernel launch settings
  const unsigned int tpb_x = 32;
  const unsigned int tpb_y = 32;
  dim3 block(tpb_x, tpb_y);
  dim3 grid((nx + tpb_x - 1)/tpb_x, (ny + tpb_y - 1)/tpb_y);

  // setup the device
  const int device = 0;
  cudaDeviceProp dev_prop;
  CHECK(cudaSetDevice(device));
  CHECK(cudaGetDeviceProperties(&dev_prop, device));

  // device arrays
  float *d_ma, *d_mb, *d_mc;
  CHECK(cudaMalloc((float **)&d_ma, n_bytes));
  CHECK(cudaMalloc((float **)&d_mb, n_bytes));
  CHECK(cudaMalloc((float **)&d_mc, n_bytes));

  CHECK(cudaMemcpy(d_ma, h_ma, n_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_mb, h_mb, n_bytes, cudaMemcpyHostToDevice));

  sumMatrixOnGPU<<< grid, block >>> (d_ma, d_mb, d_mc, nx, ny);
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_gpu, d_mc, n_bytes, cudaMemcpyDeviceToHost));

  // verify the results
  compare_float_arrays(h_ref, h_gpu, nx, ny);

  // free up the memory and reset the device
  free(h_ma); free(h_mb); free(h_ref); free(h_gpu);
  CHECK(cudaFree(d_ma)); CHECK(cudaFree(d_mb)); CHECK(cudaFree(d_mc))

  CHECK(cudaDeviceReset());

  return 0;

}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
extern "C" {
  #include "check.h"
  #include "timer.h"
  #include "init.h"
  #include "template.h"
}

void reset_host_data(float * arr, const size_t bytes) {
  memset(arr, 0, bytes);
}

float recursiveReduce(float * arr, const int n) {
  if (n == 1) return arr[0];
  // if n is odd, sum up the last element with the first element
  if ((n % 2) != 0) arr[0] += arr[n-1]; 
  int stride = n / 2;
  for (int k=0; k<stride; k++) {
    arr[k] += arr[k + stride];
  }
  return recursiveReduce(arr, stride);
}

// ########### K E R N E L S ############
__global__ void reduceNeighbored_warpDivergence(float * data, float * reduc, const int n) {
  int thx = threadIdx.x;
  int idx = thx + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  float * ptr = data + blockIdx.x * blockDim.x;

  for (int stride=1; stride < blockDim.x; stride *= 2) {
    if ( thx % (2*stride) != 0) continue;
    ptr[thx] += ptr[thx + stride];
  __syncthreads();
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

__global__ void reduceNeighbored_NoWarpDivergence(float * data, float * reduc, const int n) {
  int thx = threadIdx.x;
  int idx = thx + blockIdx.x + blockDim.x;
  if (idx >= n) return;
  float * ptr = data + blockIdx.x * blockDim.x;

  for (int stride=1; stride<blockDim.x; stride *= 2) {
    int index = 2 * stride * thx;
    if (index < blockDim.x) ptr[index] += ptr[index + stride]; // see table above
    __syncthreads();
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

__global__ void reduceInterleavedPair(float * data, float * reduc, const int n) {
  size_t thx = threadIdx.x;
  size_t idx = thx + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  float * ptr = data + blockIdx.x * blockDim.x;

  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (thx < stride) { ptr[thx] += ptr[thx + stride]; }
    __syncthreads();
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

__global__ void reduceUnroll2(float * data, float * reduc, const int n) {
  size_t thx = threadIdx.x;
  size_t idx = thx + blockIdx.x * blockDim.x * 2;
  float * ptr = data + blockIdx.x * blockDim.x * 2;

  // unrolling 2 data blocks
  if (thx + blockDim.x < n) data[idx] += data[idx + blockDim.x];
  __syncthreads();

  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride>0; stride >>= 1) {
    if (thx < stride) ptr[thx] += ptr[thx + stride];
    __syncthreads();
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

__global__ void reduceUnroll4(float * data, float * reduc, const int n) {
  size_t thx = threadIdx.x;
  size_t idx = thx + blockIdx.x * blockDim.x * 4;
  float * ptr = data + blockIdx.x * blockDim.x * 4;

  // unrolling 4 data blocks
  if (thx + blockDim.x * 3 < n) {
    data[idx] += data[idx + blockDim.x] +
                 data[idx + blockDim.x * 2] +
                 data[idx + blockDim.x * 3];
    __syncthreads();
  }

  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (thx < stride) ptr[thx] += ptr[thx + stride];
    __syncthreads();
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

__global__ void reduceUnroll8(float * data, float * reduc, const int n) {
  size_t thx = threadIdx.x;
  size_t idx = thx + blockIdx.x * blockDim.x * 8;
  float * ptr = data + blockIdx.x * blockDim.x * 8;

  // unrolling 8 data blocks
  if (thx + blockDim.x * 7 < n) {
    data[idx] += data[idx + blockDim.x] +
                 data[idx + blockDim.x * 2] + 
                 data[idx + blockDim.x * 3] +
                 data[idx + blockDim.x * 4] +
                 data[idx + blockDim.x * 5] +
                 data[idx + blockDim.x * 6] +
                 data[idx + blockDim.x * 7];
    __syncthreads();
  }

  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (thx < stride) ptr[thx] += ptr[thx + stride];
    __syncthreads();
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

__global__ void reduceUnroll8_warp(float * data, float * reduc, const int n) {
  size_t thx = threadIdx.x;
  size_t idx = thx + blockIdx.x * blockDim.x * 8;
  float * ptr = data + blockIdx.x * blockDim.x * 8;

  // unrolling 8 data blocks
  if (thx + blockDim.x * 7 < n) {
    data[idx] += data[idx + blockDim.x] +
                 data[idx + blockDim.x * 2] + 
                 data[idx + blockDim.x * 3] +
                 data[idx + blockDim.x * 4] +
                 data[idx + blockDim.x * 5] +
                 data[idx + blockDim.x * 6] +
                 data[idx + blockDim.x * 7];
    __syncthreads();
  }

  // in-place reduction in global memory
  for (int stride=blockDim.x/2; stride>32; stride>>=1) {
    if (thx < stride) ptr[thx] += ptr[thx + stride];
    __syncthreads();
  }

  // warp unrolling
  if (thx < 32) {
    volatile float * vmem = data;
    vmem[thx] += vmem[thx + 32];
    vmem[thx] += vmem[thx + 16];
    vmem[thx] += vmem[thx + 8];
    vmem[thx] += vmem[thx + 4];
    vmem[thx] += vmem[thx + 2];
    vmem[thx] += vmem[thx + 1];
  }
  if (thx == 0) reduc[blockIdx.x] = ptr[0];
}

// ########### R U N T I M E   W R A P P E R S ############
void cpu_do_sum(const float * h_data, float * h_reduc, const int n, const int m) {
  size_t nbytes = n * sizeof(float), mbytes = m * sizeof(float);
  memset(h_reduc, 0, mbytes);

  double t_cpu = getTime();
  // float * _data, * _reduc;
  float * _data  = (float *)malloc(nbytes);
  float * _reduc = (float *)malloc(mbytes);
  memcpy(_data, h_data, nbytes);
  memcpy(_reduc, h_reduc, mbytes);

  recursiveReduce(_data, n);
  _reduc[0] = _data[0];

  // free(_data); free(_reduc);
  float dt_cpu = getTime() - t_cpu;

  printf("Info: cpu_do_sum: sum_cpu = %.8f\n", _reduc[0]);
  printf("Info: dt_cpu = %.6f sec\n", dt_cpu);
}

void warmup() {
  size_t n = 1 << 14, nf = n * sizeof(float); 
  dim3 block(256, 1), grid((n+block.x-1)/block.x, 1);
  int ng = grid.x * sizeof(float);
  float * h_data  = (float *)malloc(nf);
  float * h_reduc = (float *)malloc(ng);
  for (size_t k=0; k<n; k++) h_data[k] = 1.0;
  memset(h_reduc, 0.0, ng);
  double t_gpu = getTime(), dt_gpu=0.0;
  float * d_data, * d_reduc;
  CHECK(cudaMalloc((float **)&d_data, nf));
  CHECK(cudaMalloc((float **)&d_reduc, ng));
  CHECK(cudaMemcpy(d_data, h_data, nf, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, ng, cudaMemcpyHostToDevice));
  reduceNeighbored_warpDivergence <<< grid, block >>> (d_data, d_reduc, n);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h_reduc, d_reduc, ng, cudaMemcpyDeviceToHost));
  dt_gpu = getTime() - t_gpu;
  double res_cpu = (double)n, res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: warm-up: \n"); 
  printf("Info: dt = %.6f sec\n", dt_gpu); 
  printf("Info: res: %.8f ?= %.8f \n", res_cpu, res_gpu);
  cudaFree(d_data); cudaFree(d_reduc); free(h_data); free(h_reduc);
}
/*
void gpu_no_divergence(const float * h_data, float * h_reduc,
                const int ndat, const int nred, const dim3 grid, const dim3 block) {
  size_t bytes_data  = ndat * sizeof(float);
  size_t bytes_reduc = nred * sizeof(float);
  float * d_data, * d_reduc;
  double t_gpu=0.0, dt_gpu=0.0;
  t_gpu = getTime();
  CHECK(cudaMalloc((float **)&d_data, bytes_data));
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc));
  CHECK(cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, bytes_reduc, cudaMemcpyHostToDevice));

  // kernel launch
  reduceNeighbored_NoWarpDivergence <<< grid, block >>> (d_data, d_reduc, ndat);
  CHECK(cudaDeviceSynchronize());

  // copy d2h
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost));

  dt_gpu = getTime() - t_gpu;
  double res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: gpu_no_divergence: \n");
  printf("Info: res = %.8f, dt = %.6f sec\n", res_gpu, dt_gpu);
  cudaFree(d_data); cudaFree(d_reduc);
}

void gpu_interleaved_pair(const float * h_data, float * h_reduc,
                const int ndat, const int nred, const dim3 grid, const dim3 block) {
  size_t bytes_data  = ndat * sizeof(float);
  size_t bytes_reduc = nred * sizeof(float);
  float * d_data, * d_reduc;
  double t_gpu=0.0, dt_gpu=0.0;
  t_gpu = getTime();
  CHECK(cudaMalloc((float **)&d_data, bytes_data));
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc));
  CHECK(cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, bytes_reduc, cudaMemcpyHostToDevice));

  // kernel launch
  reduceInterleavedPair <<< grid, block >>> (d_data, d_reduc, ndat);
  CHECK(cudaDeviceSynchronize());

  // copy d2h
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost));

  dt_gpu = getTime() - t_gpu; 
  double res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: gpu_interleaved_pair: \n");
  printf("Info: res = %.8f, dt = %.6f sec\n", res_gpu, dt_gpu);
}

void gpu_unroll_2(const float * h_data, float * h_reduc,
                const int ndat, const int nred, const dim3 grid, const dim3 block) {
  size_t bytes_data  = ndat * sizeof(float);
  size_t bytes_reduc = nred * sizeof(float);
  float * d_data, * d_reduc;
  double t_gpu=0.0, dt_gpu=0.0;
  t_gpu = getTime();
  CHECK(cudaMalloc((float **)&d_data, bytes_data));
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc));
  CHECK(cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, bytes_reduc, cudaMemcpyHostToDevice));

  // kernel launch
  reduceUnroll2 <<< grid, block >>> (d_data, d_reduc, ndat);
  CHECK(cudaDeviceSynchronize());

  // copy d2h
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost));

  dt_gpu = getTime() - t_gpu; 
  double res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: gpu_unroll_2: \n");
  printf("Info: res = %.8f, dt = %.6f sec\n", res_gpu, dt_gpu);
}

void gpu_unroll_4(const float * h_data, float * h_reduc,
                const int ndat, const int nred, const dim3 grid, const dim3 block) {
  size_t bytes_data  = ndat * sizeof(float);
  size_t bytes_reduc = nred * sizeof(float);
  float * d_data, * d_reduc;
  double t_gpu=0.0, dt_gpu=0.0;
  t_gpu = getTime();
  CHECK(cudaMalloc((float **)&d_data, bytes_data));
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc));
  CHECK(cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, bytes_reduc, cudaMemcpyHostToDevice));

  // kernel launch
  reduceUnroll4 <<< grid, block >>> (d_data, d_reduc, ndat);
  CHECK(cudaDeviceSynchronize());

  // copy d2h
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost));

  dt_gpu = getTime() - t_gpu; 
  double res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: gpu_unroll_4: \n");
  printf("Info: res = %.8f, dt = %.6f sec\n", res_gpu, dt_gpu);
}

void gpu_unroll_8(const float * h_data, float * h_reduc,
                const int ndat, const int nred, const dim3 grid, const dim3 block) {
  size_t bytes_data  = ndat * sizeof(float);
  size_t bytes_reduc = nred * sizeof(float);
  float * d_data, * d_reduc;
  double t_gpu=0.0, dt_gpu=0.0;
  t_gpu = getTime();
  CHECK(cudaMalloc((float **)&d_data, bytes_data));
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc));
  CHECK(cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, bytes_reduc, cudaMemcpyHostToDevice));

  // kernel launch
  reduceUnroll8 <<< grid, block >>> (d_data, d_reduc, ndat);
  CHECK(cudaDeviceSynchronize());

  // copy d2h
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost));

  dt_gpu = getTime() - t_gpu; 
  double res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: gpu_unroll_2: \n");
  printf("Info: res = %.8f, dt = %.6f sec\n", res_gpu, dt_gpu);
}

void gpu_unroll_8_warp_unroll(const float * h_data, float * h_reduc,
                const int ndat, const int nred, const dim3 grid, const dim3 block) {
  size_t bytes_data  = ndat * sizeof(float);
  size_t bytes_reduc = nred * sizeof(float);
  float * d_data, * d_reduc;
  double t_gpu=0.0, dt_gpu=0.0;
  t_gpu = getTime();
  CHECK(cudaMalloc((float **)&d_data, bytes_data));
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc));
  CHECK(cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_reduc, h_reduc, bytes_reduc, cudaMemcpyHostToDevice));

  // kernel launch
  reduceUnroll8_warp <<< grid, block >>> (d_data, d_reduc, ndat);
  CHECK(cudaDeviceSynchronize());

  // copy d2h
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost));

  dt_gpu = getTime() - t_gpu; 
  double res_gpu = 0.0;
  for (int k=0; k<grid.x; k++) res_gpu += h_reduc[k];
  printf("Info: gpu_unroll_2: \n");
  printf("Info: res = %.8f, dt = %.6f sec\n", res_gpu, dt_gpu);
}
*/
// ########### M A I N ###########
int main(int argc, char** argv) {

  // test input and execution 
  if (argc != 2) {
    printf("Error: Wrong number of arguments. Call e.g. \n");
    printf("  $> run.exe <threads_x>\n");
    printf("where <threads_x> is the number of threads \n");
    printf("per block along x-direction.\n");
    exit(EXIT_FAILURE);
  }

  const int dev = 0;
  cudaDeviceProp dev_prop;
  CHECK(cudaGetDeviceProperties(&dev_prop, dev));
  CHECK(cudaSetDevice(dev));
  printf("Info: Using device: %s. \n", dev_prop.name);

  // problem setup
  const unsigned int size = 1 << 24;

  // kernel launch configs
  int thrdx = atoi(argv[1]);
  if (thrdx < 1) thrdx = 128;
  dim3 block(thrdx, 1);
  dim3 grid((size-1)/block.x+1, 1);
  printf("Info: grid=(%d, %d), block=(%d, %d)\n", grid.x, grid.y, block.x, block.y);

  const unsigned int bytes_data  = size * sizeof(float);
  const unsigned int bytes_reduc = grid.x * sizeof(float);

  // host arrays
  float * h_data  = (float *) malloc(bytes_data);   // length N
  float * h_reduc = (float *) malloc(bytes_reduc);  // length num. grid blocks

  // initialize data to random values
  init_float(h_data, size);
  reset_host_data(h_reduc, bytes_reduc);

  cpu_do_sum(h_data, h_reduc, size, grid.x);

  // warm up the device
  warmup();

  // warp divergence
  printf("\nInfo: reduceNeighbored_warpDivergence \n"); 
  TestPlatform(reduceNeighbored_warpDivergence, \
               h_data, h_reduc, size, grid.x, grid, block);

  // no warp divergence
  printf("\nInfo: reduceNeighbored_NoWarpDivergence \n"); 
  TestPlatform(reduceNeighbored_NoWarpDivergence, \
               h_data, h_reduc, size, grid.x, grid, block);

  // interleaved-pair approach
  printf("\nInfo: reduceInterleavedPair \n"); 
  TestPlatform(reduceInterleavedPair, \
               h_data, h_reduc, size, grid.x, grid, block); 

  // loop unrolling
  // 2 blocks per thread
  printf("\nInfo: reduceUnroll2\n");
  dim3 grid_2((size-1)/block.x+1, 1);
  TestPlatform(reduceUnroll2, h_data, h_reduc, size, grid_2.x, grid_2, block);

  // 4 blocks per thread
  printf("\nInfo: reduceUnroll4\n");
  dim3 grid_4((size-1)/block.x+1, 1);
  TestPlatform(reduceUnroll4, h_data, h_reduc, size, grid_4.x, grid_4, block);

  // 8 blocks per thread
  printf("\nInfo: reduceUnroll8\n");
  dim3 grid_8((size-1)/block.x+1, 1);
  TestPlatform(reduceUnroll8, h_data, h_reduc, size, grid_8.x, grid_8, block);

  // 8 blocks per thread incuding warp unrolling
  printf("\nInfo: reduceUnroll8_warp\n");
  TestPlatform(reduceUnroll8_warp, \
               h_data, h_reduc, size, grid_8.x, grid_8, block);

  // free up memory
  free(h_data);
  free(h_reduc);

  // thanks and goodbye :-)
  CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}

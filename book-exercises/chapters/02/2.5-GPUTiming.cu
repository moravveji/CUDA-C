#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(cmnd) {                  \
  cudaError_t ierr = cmnd;                     \
  if (ierr != cudaSuccess) {           \
    printf("Error: %s:%d: ", __FILE__, __LINE__, cudaGetErrorString(ierr)); \
    exit(ierr);                        \
  }                                    \
}

void initData(float * arr, const int n) {
  time_t t;
  srand((unsigned int) time(&t));
  for (int k=0; k<n; k++) arr[k] = (float)(rand() & 0xFF) / 10.0;
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)tp.tv_sec + (double)tp.tv_usec*1.0e-6;
}

void Add_on_host(const float * a, const float * b, float * c, const int n) {
  for (int k=0; k<n; k++) c[k] = a[k] + b[k];
}

__global__ void Add_on_device(const float * a, const float * b, float * c, const int n) {
  size_t k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k < n) c[k] = a[k] + b[k];
}

void check_result(const float *a, const float *b, const int n){
  const double epsilon = 1.0e-8;
  double diff = 0.0;
  bool match = 1;
  for (int k=0; k<n; k++){
    diff = abs(a[k] - b[k]);
    if (diff > epsilon){
      match = 0;
      printf("Error: check_result: diff=%16.12f at k=%d\n", diff, k);
      break;
    }
  }
  if (match) printf("Success: all elements match better than epsilon=%16.12f\n", epsilon);
}

int main(int argc, char ** argv) {
  printf("Info: Starting %s ... \n", argv[0]);

  // problem sizes and kernel configs
  const int n_elem = 1 << 24;
  const size_t n_byte = n_elem * sizeof(float);
  const int tpb_x = 128;
  dim3 tpb(tpb_x, 1, 1);
  dim3 nblocks((n_elem + tpb_x - 1) / tpb_x, 1, 1); 

  // timing
  double t0, dt_host, dt_gpu, dt_h2d, dt_kern, dt_d2h;

  // addition on host
  t0 = cpuSecond();
  float *h_a, *h_b, *h_ref; //, *d_ref;
  h_a = (float *)malloc(n_byte);
  h_b = (float *)malloc(n_byte);
  h_ref = (float *)malloc(n_byte);  // reference result from host
  // d_ref = (float *)malloc(n_byte);  // reference result from device

  initData(h_a, n_elem);
  initData(h_b, n_elem);

  memset(h_a, 0, n_byte);
  memset(h_b, 0, n_byte);

  Add_on_host(h_a, h_b, h_ref, n_elem);
  dt_host = cpuSecond() - t0;

  // device addition
  const int dev = 0;
  cudaDeviceProp dev_prop;
  CHECK(cudaSetDevice(dev));
  printf("Info: device #%d is: %s\n", dev, dev_prop.name);

  t0 = cpuSecond();
  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc((float **)&d_a, n_byte));
  CHECK(cudaMalloc((float **)&d_b, n_byte));
  CHECK(cudaMalloc((float **)&d_c, n_byte));

  CHECK(cudaMemcpy(d_a, h_a, n_byte, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, n_byte, cudaMemcpyHostToDevice));

  dt_h2d = cpuSecond() - t0;

  // Kernel launch
  t0 = cpuSecond();
  Add_on_device<<<nblocks, tpb>>>(d_a, d_b, d_c, n_elem);
  CHECK(cudaDeviceSynchronize());
  dt_kern = cpuSecond() - t0;

  float * h_res;
  h_res = (float *)malloc(n_byte);
  t0 = cpuSecond();
  CHECK(cudaMemcpy(h_res, d_c, n_byte, cudaMemcpyDeviceToHost));
  dt_d2h = cpuSecond() - t0;

  check_result(h_ref, h_res, n_elem);
  // {
  //   const double epsilon = 1.0e-8;
  //   double diff = 0.0;
  //   bool match = 1;
  //   for (int k=0; k<n_elem; k++){
  //     diff = abs(h_ref[k] - d_ref[k]);
  //     if (diff > epsilon){
  //       match = 0;
  //       printf("Error: check_result: diff=%16.12f at k=%d\n", diff, k);
  //       break;
  //     }
  //   }
  //   if (match) printf("Success: all elements match better than epsilon=%16.12f\n", epsilon);  
  // }
  
  dt_gpu = dt_h2d + dt_kern + dt_d2h;
  printf("\n%s\n", "Timing results ...");
  printf("dt_host: %12.8f (sec)\n", dt_host);
  printf("dt_h2d:  %12.8f (sec)\n", dt_h2d);
  printf("dt_kern: %12.8f (sec)\n", dt_kern);
  printf("dt_d2h:  %12.8f (sec)\n", dt_d2h);
  printf("dt_gpu:  %12.8f (sec)\n", dt_gpu);
  printf("dt_host / dt_gpu = %6.2f \n", dt_host / dt_gpu);
  printf("\n");

  // Free up the memory on host and device
  free(h_a); free(h_b); free(h_ref); free(h_res);
  CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));

  return 0;
}

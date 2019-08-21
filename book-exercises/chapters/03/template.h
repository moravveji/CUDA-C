
#define TestPlatform(KERNEL, \
                     H_DATA, \
                     H_REDUCE, \
                     N_DATA, \
                     N_REDUCE, \
                     GRID, \
                     BLOCK) {\
  size_t bytes_data  = N_DATA * sizeof(float); \
  size_t bytes_reduc = N_REDUCE * sizeof(float); \
  float * d_data, * d_reduc; \
  double t_gpu=0.0, dt_gpu=0.0; \
  \
  t_gpu = getTime(); \
  CHECK(cudaMalloc((float **)&d_data, bytes_data)); \
  CHECK(cudaMalloc((float **)&d_reduc, bytes_reduc)); \
  CHECK(cudaMemcpy(d_data, H_DATA, bytes_data, cudaMemcpyHostToDevice)); \
  CHECK(cudaMemcpy(d_reduc, H_REDUCE, bytes_reduc, cudaMemcpyHostToDevice)); \
  \
  KERNEL <<< GRID, BLOCK >>> (d_data, d_reduc, N_DATA); \
  CHECK(cudaDeviceSynchronize()); \
  \
  CHECK(cudaMemcpy(h_reduc, d_reduc, bytes_reduc, cudaMemcpyDeviceToHost)); \
  \
  dt_gpu = getTime() - t_gpu; \
  double res_gpu = 0.0; \
  for (int k=1; k<grid.x; k++) res_gpu += h_reduc[k]; \
  printf("Info: res = %.8f; dt = %.6f sec\n", res_gpu, dt_gpu); \
  memset(H_REDUCE, 0, bytes_reduc); \
}

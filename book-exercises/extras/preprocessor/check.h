
#ifndef _CHECK_H
#define _CHECK_H
  #include <stdio.h>

  #define CHECK(cmnd) {        \
    cudaError_t err = cmnd;    \
    if (err != cudaSuccess) {  \
      printf("Error: %s(%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(err);                \
    }                          \
  }

#endif

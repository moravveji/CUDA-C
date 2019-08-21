#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include "compare_arrays.h"

void compare_float_arrays(const float * a, const float * b, const int nx, const int ny) {
  const double epsilon = 1.0e-8;
  double diff = 0.0;
  bool match = 1;
  for (size_t iy=0; iy<ny; iy++) {
    for (size_t ix=0; ix<nx; ix++) {
      diff = abs(a[ix] - b[ix]);
      if (diff > epsilon) {
        printf("%s\n", "Error: Result mismatch: diff=%16.12f > epsilon=%16.12f \n", diff, epsilon);
        match = 0;
        break;
      }
    }
    a += nx; b += nx;
  }
  if (match) printf("Info: compare_float_arrays: all elements match!\n");
}


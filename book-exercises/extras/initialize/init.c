#include <time.h>
#include <stdlib.h>
#include <math.h>

void init_float(float * a, int const n) {
  time_t t;
  srand( (unsigned int) time(&t) );
  int sign = 1;
  float divisor = (float)RAND_MAX;
  for (size_t k=0; k<n; k++) {
    sign = (int) pow(-1, k % 2);
    a[k] = sign * (float)(rand() & 0xFF) / divisor;
  }
}

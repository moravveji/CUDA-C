#include <stdlib.h>
#include <sys/time.h>

double getTime() {
  struct timeval now;
  gettimeofday(&now, NULL);
  return (double)now.tv_sec + (double)now.tv_usec*1e-6;
}


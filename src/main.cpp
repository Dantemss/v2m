#include <cstdio>

#include "parallel_cv.hpp"
#include "dense_flow.hpp"

using namespace parallel_cv;

#define VIDEO_PATH "videos/video.mpeg"
#define NUM_WORKER_THREADS 8
#define WORK_FUNCTION denseFlow

int main(void) {
  run(VIDEO_PATH, NUM_WORKER_THREADS, WORK_FUNCTION);

  return 0;
}

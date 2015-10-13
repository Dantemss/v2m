#include <cstdio>

#include "parallel_cv.hpp"
#include "parallel_cv/command/dense_flow.hpp"

using namespace parallel_cv;

#define VIDEO_PATH "videos/video.mpeg"
#define NUM_WORKER_THREADS 8
#define COMMAND_FUNCTION command::denseFlow

int main(void) {
  run(VIDEO_PATH, NUM_WORKER_THREADS, COMMAND_FUNCTION);

  return 0;
}

#include <cstdio>

#include "parallel_cv.hpp"
#include "parallel_cv/command/dense_flow.hpp"

using namespace parallel_cv;

#define NUM_WORKER_THREADS 8
#define COMMAND_FUNCTION command::denseFlow

int main(int argc, char *argv[]) {
  cv::String video_path;

  switch(argc) {
    case 1: video_path = "0";
            break;
    case 2: video_path = argv[1];
            break;
    default: log("Error: too many arguments");
             return 1;
  }

  run(video_path, NUM_WORKER_THREADS, COMMAND_FUNCTION);

  return 0;
}

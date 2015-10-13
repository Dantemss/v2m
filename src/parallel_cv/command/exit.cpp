#include <pthread.h>

#include "exit.hpp"

using namespace cv;

namespace parallel_cv {
  namespace command {
    Mat exit(Mat& frame, Mat& prev) {
      pthread_exit(NULL);
      return frame;
    }
  }
}

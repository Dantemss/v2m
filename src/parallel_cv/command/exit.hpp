#ifndef PARALLEL_CV_COMMAND_EXIT_HPP
#define PARALLEL_CV_COMMAND_EXIT_HPP

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  namespace command {
    cv::Mat exit(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

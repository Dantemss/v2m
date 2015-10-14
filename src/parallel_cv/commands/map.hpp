#ifndef PARALLEL_CV_COMMAND_MAP_HPP
#define PARALLEL_CV_COMMAND_MAP_HPP

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  namespace commands {
    cv::Mat map(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

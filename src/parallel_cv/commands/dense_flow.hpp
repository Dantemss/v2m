#ifndef PARALLEL_CV_COMMAND_DENSE_FLOW_HPP
#define PARALLEL_CV_COMMAND_DENSE_FLOW_HPP

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  namespace commands {
    cv::Mat denseFlow(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

#ifndef PARALLEL_CV_COMMANDS_EXIT_HPP
#define PARALLEL_CV_COMMANDS_EXIT_HPP

#include <opencv2/core/mat.hpp>

namespace parallel_cv {
  namespace commands {
    cv::Mat exit(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

#ifndef V2M_COMMANDS_EXIT_HPP
#define V2M_COMMANDS_EXIT_HPP

#include <opencv2/core/mat.hpp>

namespace v2m {
  namespace commands {
    cv::Mat exit(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

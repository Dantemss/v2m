#ifndef V2M_COMMANDS_MAP_HPP
#define V2M_COMMANDS_MAP_HPP

#include <opencv2/core/mat.hpp>

namespace v2m {
  namespace commands {
    cv::Mat map(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

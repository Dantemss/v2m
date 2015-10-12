#ifndef WORK_HPP
#define WORK_HPP

#include "opencv2/core/mat.hpp"

namespace Work {
  void *work(void *arg);

  void *output(void *arg);

  cv::Mat exit(cv::Mat& frame, cv::Mat& prev);
}

#endif

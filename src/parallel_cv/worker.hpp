#ifndef PARALLEL_CV_WORKER_HPP
#define PARALLEL_CV_WORKER_HPP

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  namespace worker {
    void *work(void *arg);

    void *output(void *arg);

    cv::Mat exit(cv::Mat& frame, cv::Mat& prev);
  }
}

#endif

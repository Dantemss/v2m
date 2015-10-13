#ifndef PARALLEL_CV_HPP
#define PARALLEL_CV_HPP

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  void run(cv::String video_path,
           int num_worker_threads,
           cv::Mat (*const work_function)(cv::Mat&, cv::Mat&));
}

#endif

#ifndef PARALLEL_CV_HPP
#define PARALLEL_CV_HPP

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  void log(const char* message);

  void run(cv::String video_path,
           size_t num_worker_threads,
           cv::Mat (*const command_function)(cv::Mat&, cv::Mat&));
}

#endif
#ifndef PARALLEL_CV_COMMAND_DENSE_FLOW_HPP
#define PARALLEL_CV_COMMAND_DENSE_FLOW_HPP

#include <vector>

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  namespace commands {
    namespace dense_flow {
      cv::Mat_<cv::Point2f> calc(cv::Mat& frame, cv::Mat& prev);

      std::vector< cv::Ptr< cv::Vec<double, 8> > > getFeatures(cv::Mat_<cv::Point2f>& flow);
    }
  }
}

#endif

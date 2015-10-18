#ifndef PARALLEL_CV_COMMANDS_ALGORITHMS_SPARSE_FLOW_HPP
#define PARALLEL_CV_COMMANDS_ALGORITHMS_SPARSE_FLOW_HPP

#include <vector>

#include "opencv2/core/mat.hpp"

namespace parallel_cv {
  namespace commands {
    namespace algorithms {
      namespace sparse_flow {
        std::vector< cv::Vec<double, 8> > get3dFlow(cv::Mat& frame, cv::Mat& prev);
      }
    }
  }
}

#endif

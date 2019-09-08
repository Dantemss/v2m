#include <opencv2/imgproc/imgproc.hpp>

#include "map.hpp"

#include "algorithms/dense_flow.hpp"

namespace parallel_cv {
  namespace commands {
    cv::Mat map(cv::Mat& frame, cv::Mat& prev) {
      if (prev.empty()) return frame;

      cv::Mat gray, prev_gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::cvtColor(prev, prev_gray, cv::COLOR_BGR2GRAY);

      std::vector< cv::Vec<double, 8> > flow = algorithms::dense_flow::get3dFlow(gray, prev_gray);

      std::vector<int> labels;

      cv::Mat_<cv::Vec3b> output(frame.size(), 0.0);
      size_t i;
      for(i = 0; i < flow.size(); i++) {
        cv::Vec<double, 8> fl = flow[i];
        int x = fl[0];
        int y = fl[1];
        output(x, y) = cv::Vec3b(fl[4], frame.at<cv::Vec3b>(x, y)[1], -fl[4]);
      }

      return output;
    }
  }
}

#include <opencv2/imgproc/imgproc.hpp>

#include "map.hpp"

#include "algorithms/dense_flow.hpp"

using namespace std;

namespace parallel_cv {
  namespace commands {
    cv::Mat map(cv::Mat& frame, cv::Mat& prev) {
      if (prev.empty()) return frame;

      cv::Mat gray, prev_gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::cvtColor(prev, prev_gray, cv::COLOR_BGR2GRAY);

      std::vector<cv::Vec<double, 8>> flow = algorithms::dense_flow::get3dFlow(gray, prev_gray);

      size_t flow_size = flow.size();
      double tx(0);
      double ty(0);
      double tz(0);
      cv::Mat_<cv::Vec3b> output(frame.size(), 0.0);

      for(size_t i = 0; i < flow_size; i++) {
        cv::Vec<double, 8> fl = flow[i];
        int x = fl[0];
        int y = fl[1];
        tx += fl[2];
        ty += fl[3];
        double scaled_vz = fl[4]*32;
        tz += scaled_vz;
        double output_value = scaled_vz*4;
        output(x, y) = cv::Vec3b(
          min(max( output_value, 0.), 255.),
          frame.at<cv::Vec3b>(x, y)[1],
          min(max(-output_value, 0.), 255.)
        );
      }

      printf("VX: %f, VY: %f, VZ: %f\n", tx/flow_size, ty/flow_size, tz/flow_size);

      return output;
    }
  }
}

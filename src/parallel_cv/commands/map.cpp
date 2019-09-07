#include <opencv2/imgproc/imgproc.hpp>

#include "map.hpp"

#include "../../parallel_cv.hpp"
#include "algorithms/dense_flow.hpp"

using namespace cv;

#define MAX_CLUSTER_DISTANCE  0.01

namespace parallel_cv {
  namespace commands {
    bool cluster(const Vec<double, 8>& a, const Vec<double, 8>& b) {
      int i;
      double dist = 0;
      for (i = 2; i < 8; i++) {
        dist += pow((a[i] - b[i]), 2);
      }

      return dist < MAX_CLUSTER_DISTANCE;
    }

    /**
     * @function map
     */
    Mat map(Mat& frame, Mat& prev) {
      if (prev.empty()) return frame;

      Mat gray, prev_gray;
      cvtColor(frame, gray, COLOR_BGR2GRAY);
      cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

      std::vector< Vec<double, 8> > flow = algorithms::dense_flow::get3dFlow(gray, prev_gray);

      std::vector<int> labels;
      //partition(flow, labels, &cluster);

      Mat_<Vec3b> output(frame.size(), 0.0);
      size_t i;
      for(i = 0; i < flow.size(); i++) {
        Vec<double, 8> fl = flow[i];
        int x = fl[0];
        int y = fl[1];
        output(x, y) = Vec3b(fl[4], frame.at<Vec3b>(x, y)[1], -fl[4]);
      }

      return output;
    }
  }
}

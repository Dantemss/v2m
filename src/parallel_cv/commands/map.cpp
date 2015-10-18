#include "opencv2/imgproc/imgproc.hpp"

#include "map.hpp"

#ifdef DENSE_FLOW
  #include "algorithms/dense_flow.hpp"
#else
  #include "algorithms/sparse_flow.hpp"
#endif

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
      cvtColor(frame, gray, CV_BGR2GRAY);
      cvtColor(prev, prev_gray, CV_BGR2GRAY);

      #ifdef DENSE_FLOW
        std::vector< Vec<double, 8> > flow = algorithms::dense_flow::get3dFlow(gray, prev_gray);
      #else
        std::vector< Vec<double, 8> > flow = algorithms::sparse_flow::get3dFlow(gray, prev_gray);
      #endif

      std::vector<int> labels;
      partition(flow, labels, &cluster);

      #ifdef DENSE_FLOW
        Mat_<Vec3b> output(frame.size(), 0.0);
      #else
        Mat_<Vec3b> output(frame);
      #endif
      size_t i;
      for(i = 0; i < flow.size(); i++) {
        #ifdef DENSE_FLOW
          output(flow[i][0], flow[i][1]) = Vec3b(labels[i]/(256*256),
                                                (labels[i]/256) % 256,
                                                 labels[i] % 256);
        #else
          circle(output, Point(flow[i][1], flow[i][0]), 3, Scalar(0,255,0), -1, 8);
        #endif
      }

      return output;
    }
  }
}

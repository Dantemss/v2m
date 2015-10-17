#include "opencv2/imgproc/imgproc.hpp"

#include "map.hpp"
#include "algorithms/dense_flow.hpp"

using namespace cv;

#define INITIAL_INDEX 2
#define MAX_DISTANCE  0.01

namespace parallel_cv {
  namespace commands {
    bool cluster(const Ptr< Vec<double, 8> >& a, const Ptr< Vec<double, 8> >& b) {
      Vec<double, 8> va, vb;
      va = *a;
      vb = *b;

      int i;
      double dist = 0;
      for (i = INITIAL_INDEX; i < 8; i++) {
        dist += pow((va[i] - vb[i]), 2);
      }

      return dist < MAX_DISTANCE;
    }

    /**
     * @function map
     */
    Mat map(Mat& frame, Mat& prev) {
      if (prev.empty()) return frame;

      Mat gray, prev_gray;
      cvtColor(frame, gray, CV_BGR2GRAY);
      cvtColor(prev, prev_gray, CV_BGR2GRAY);

      Mat_<Point2f> flow = algorithms::dense_flow::calc(gray, prev_gray);
      std::vector< Ptr< Vec<double, 8> > > features = algorithms::dense_flow::getFeatures(flow);

      std::vector<int> labels;
      partition(features, labels, &cluster);

      Mat_<Vec3b> output(flow.size(), 0);
      size_t i;
      for(i = 0; i < features.size(); i++) {
        output((*features[i])[0], (*features[i])[1]) = Vec3b(labels[i]/(256*256),
                                                            (labels[i]/256) % 256,
                                                             labels[i] % 256);
      }

      return output;
    }
  }
}

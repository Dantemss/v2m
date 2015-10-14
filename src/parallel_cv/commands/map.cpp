#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include "map.hpp"

using namespace cv;

namespace parallel_cv {
  namespace commands {
    inline bool isFlowCorrect(Point2f u) {
      return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
    }

    static void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1) {
      dst = Mat::zeros(flow.size(), CV_8UC1);

      // determine motion range:
      float maxrad = maxmotion;

      if (maxmotion <= 0) {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y) {
          for (int x = 0; x < flow.cols; ++x) {
            Point2f u = flow(y, x);

            if (!isFlowCorrect(u)) continue;

            maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
          }
        }
      }

      for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
          Point2f u = flow(y, x);

          if (isFlowCorrect(u) && (u.x > 0.1 || u.y > 0.1)) dst.at<uchar>(y, x) = 1;
        }
      }
    }

    /**
     * @function map
     */
    Mat map(Mat& frame, Mat& prev) {
      if (prev.empty()) return frame;

      Mat gray, prev_gray;
      cvtColor(frame, gray, CV_BGR2GRAY);
      cvtColor(prev, prev_gray, CV_BGR2GRAY);

      Mat_<Point2f> flow;
      calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 3, 20, 3, 5, 1.1, 0);
      //Ptr<DualTVL1OpticalFlow> tvl1 = createOptFlow_DualTVL1();
      //tvl1->calc(prev_gray, gray, flow);

      Mat flow_out, out;
      drawOpticalFlow(flow, flow_out);
      out = Scalar::all(0);
      frame.copyTo(out, flow_out);

      return out;
    }
  }
}

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include "map.hpp"

using namespace cv;

#define MAX_FLOW 25
#define K_SCALE 0.25*(2 - sqrt(2))
#define QRTR_SQRT_2_SCALED 0.125*(sqrt(2) - 1)

namespace parallel_cv {
  namespace commands {
    Matx33d div_vx_kernel(QRTR_SQRT_2_SCALED, 0, -QRTR_SQRT_2_SCALED,
                          K_SCALE,            0, -K_SCALE,
                          QRTR_SQRT_2_SCALED, 0, -QRTR_SQRT_2_SCALED);
    Matx33d div_vy_kernel(QRTR_SQRT_2_SCALED,  K_SCALE,  QRTR_SQRT_2_SCALED,
                          0,                   0,        0,
                          -QRTR_SQRT_2_SCALED, -K_SCALE, -QRTR_SQRT_2_SCALED);

    Matx33d curlz_vx_kernel(QRTR_SQRT_2_SCALED,  K_SCALE,  QRTR_SQRT_2_SCALED,
                            0,                   0,        0,
                            -QRTR_SQRT_2_SCALED, -K_SCALE, -QRTR_SQRT_2_SCALED);
    Matx33d curlz_vy_kernel(-QRTR_SQRT_2_SCALED, 0, QRTR_SQRT_2_SCALED,
                            -K_SCALE,            0, K_SCALE, 
                            -QRTR_SQRT_2_SCALED, 0, QRTR_SQRT_2_SCALED);

    Matx13d curly_vz_kernel(0.5, 0, -0.5);
    Matx31d curlx_vz_kernel(-0.5,
                            0,
                            0.5);

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

      int i, j;
      Point2f p;
      Size flow_size = flow.size();
      Mat_<double> vx(flow_size), vy(flow_size), vz(flow_size),
                   curlx(flow_size), curly(flow_size), curlz(flow_size),
                   temp_x(flow_size), temp_y(flow_size);
      for (i = 0; i < flow_size.height; i++) {
        for (j = 0; j < flow_size.width; j++) {
          p = flow(i, j);

          if (cvIsNaN(p.x) || cvIsNaN(p.y) || abs(p.x) > MAX_FLOW || abs(p.y) > MAX_FLOW) continue;

          vx(i, j)  = p.x;
          vy(i, j)  = p.y;
        }
      }

      filter2D(vx, temp_x, -1, div_vx_kernel);
      filter2D(vy, temp_y, -1, div_vy_kernel);
      vz = temp_x + temp_y;

      filter2D(vx, temp_x, -1, curlz_vx_kernel);
      filter2D(vy, temp_y, -1, curlz_vy_kernel);
      curlz = temp_x + temp_y;

      filter2D(vz, curly, -1, curly_vz_kernel);
      filter2D(vz, curlx, -1, curlx_vz_kernel);

      return vz + 0.5;
    }
  }
}

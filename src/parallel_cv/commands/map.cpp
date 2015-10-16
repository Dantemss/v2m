#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include "map.hpp"

using namespace cv;

#define MAX_FLOW 25

#define K_SCALE            0.25*(2 - sqrt(2))
#define QRTR_SQRT_2_SCALED 0.125*(sqrt(2) - 1)

#define TVL1

#define FB_PYR_SCALE  0.5
#define FB_LEVELS     3
#define FB_WINSIZE    20
#define FB_ITERATIONS 3
#define FB_POLY_N     7
#define FB_POLY_SIGMA 1.5
#define FB_FLAGS      OPTFLOW_FARNEBACK_GAUSSIAN

#define TVL1_TAU              0.25
#define TVL1_LAMBDA           0.15
#define TVL1_THETA            0.3
#define TVL1_GAMMA            0.0
#define TVL1_SCALES_NUMBER    1
#define TVL1_WARPINGS_NUMBER  1
#define TVL1_EPSILON          0.1
#define TVL1_INNER_ITERATIONS 30
#define TVL1_OUTER_ITERATIONS 10
#define TVL1_USE_INITIAL_FLOW false
#define TVL1_SCALE_STEP       0.8
#define TVL1_MEDIAN_FILTERING 5

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
      #ifdef FARNEBACK
        calcOpticalFlowFarneback(
          prev_gray, gray, flow,
          FB_PYR_SCALE, FB_LEVELS, FB_WINSIZE, FB_ITERATIONS, FB_POLY_N, FB_POLY_SIGMA, FB_FLAGS
        );
      #endif

      #ifdef TVL1
        Ptr<DualTVL1OpticalFlow> tvl1 = createOptFlow_DualTVL1();
        tvl1->setTau(TVL1_TAU);
        tvl1->setLambda(TVL1_LAMBDA);
        tvl1->setTheta(TVL1_THETA);
        tvl1->setGamma(TVL1_GAMMA);
        tvl1->setScalesNumber(TVL1_SCALES_NUMBER);
        tvl1->setWarpingsNumber(TVL1_WARPINGS_NUMBER);
        tvl1->setEpsilon(TVL1_EPSILON);
        tvl1->setInnerIterations(TVL1_INNER_ITERATIONS);
        tvl1->setOuterIterations(TVL1_OUTER_ITERATIONS);
        tvl1->setUseInitialFlow(TVL1_USE_INITIAL_FLOW);
        tvl1->setScaleStep(TVL1_SCALE_STEP);
        tvl1->setMedianFiltering(TVL1_MEDIAN_FILTERING);
        tvl1->calc(prev_gray, gray, flow);
      #endif


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

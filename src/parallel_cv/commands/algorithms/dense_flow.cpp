#include "opencv2/video/tracking.hpp"

#include "dense_flow.hpp"

using namespace cv;

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

#define MAX_FLOW      10

#define K_ADJACENT    0.25
#define K_CORNER      0.125

namespace parallel_cv {
  namespace commands {
    namespace algorithms {
      namespace dense_flow {
        Matx33d div_vx_kernel(K_CORNER,   0, -K_CORNER,
                              K_ADJACENT, 0, -K_ADJACENT,
                              K_CORNER,   0, -K_CORNER);
        Matx33d div_vy_kernel( K_CORNER,  K_ADJACENT,  K_CORNER,
                               0,         0,           0,
                              -K_CORNER, -K_ADJACENT, -K_CORNER);

        Matx33d curlz_vx_kernel( K_CORNER,  K_ADJACENT,  K_CORNER,
                                 0,         0,           0,
                                -K_CORNER, -K_ADJACENT, -K_CORNER);
        Matx33d curlz_vy_kernel(-K_CORNER,   0, K_CORNER,
                                -K_ADJACENT, 0, K_ADJACENT,
                                -K_CORNER,   0, K_CORNER);

        Matx13d curly_vz_kernel(0.5, 0, -0.5);
        Matx31d curlx_vz_kernel(-0.5,
                                 0,
                                 0.5);

        Mat_<Point2f> calc(Mat& frame, Mat& prev) {
          Mat_<Point2f> flow;

          #ifdef FARNEBACK
            calcOpticalFlowFarneback(
              prev, frame, flow,
              FB_PYR_SCALE, FB_LEVELS, FB_WINSIZE,
              FB_ITERATIONS, FB_POLY_N, FB_POLY_SIGMA, FB_FLAGS
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
            tvl1->calc(prev, frame, flow);
          #endif

          return flow;
        } 

        std::vector< Ptr< Vec<double, 8> > > getFeatures(Mat_<Point2f>& flow) {
          int i, j;
          Point2f p;
          Size flow_size = flow.size();
          Mat_<double> vx(flow_size, 0.0), vy(flow_size, 0.0), vz(flow_size, 0.0),
                       curlx(flow_size, 0.0), curly(flow_size, 0.0), curlz(flow_size, 0.0),
                       temp_x(flow_size, 0.0), temp_y(flow_size, 0.0);
          for (i = 0; i < flow_size.height; i++) {
            for (j = 0; j < flow_size.width; j++) {
              p = flow(i, j);

              if (cvIsNaN(p.x) || cvIsNaN(p.y) || abs(p.x) > MAX_FLOW || abs(p.y) > MAX_FLOW) {
                continue;
              }

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

          std::vector< Ptr< Vec<double, 8> > > features;
          for (i = 0; i < flow_size.height; i++) {
            for (j = 0; j < flow_size.width; j++) {
              features.push_back(makePtr< Vec<double, 8> >(
                i, j, vx(i, j), vy(i, j), vz(i, j), curlz(i, j), curly(i, j), curlx(i, j)
              ));
            }
          }

          return features;
        }
      }
    }
  }
}

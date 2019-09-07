#include <opencv2/video/tracking.hpp>

#include "dense_flow.hpp"

using namespace cv;

#define MAX_FLOW      10

#define K_ADJACENT    0.25
#define K_CORNER      0.125

namespace parallel_cv {
  namespace commands {
    namespace algorithms {
      namespace dense_flow {
        const Matx33d div_vx_kernel(K_CORNER,   0, -K_CORNER,
                                    K_ADJACENT, 0, -K_ADJACENT,
                                    K_CORNER,   0, -K_CORNER);
        const Matx33d div_vy_kernel( K_CORNER,  K_ADJACENT,  K_CORNER,
                                     0,         0,           0,
                                    -K_CORNER, -K_ADJACENT, -K_CORNER);

        const Matx33d curlz_vx_kernel( K_CORNER,  K_ADJACENT,  K_CORNER,
                                       0,         0,           0,
                                      -K_CORNER, -K_ADJACENT, -K_CORNER);
        const Matx33d curlz_vy_kernel(-K_CORNER,   0, K_CORNER,
                                      -K_ADJACENT, 0, K_ADJACENT,
                                      -K_CORNER,   0, K_CORNER);

        const Matx13d curly_vz_kernel(0.5, 0, -0.5);
        const Matx31d curlx_vz_kernel(-0.5,
                                       0,
                                       0.5);

        inline Mat_<Point2f> getFlow(Mat& frame, Mat& prev) {
          Mat_<Point2f> flow;

          Ptr<DenseOpticalFlow> dof = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
          dof->calc(prev, frame, flow);

          return flow;
        }

        std::vector< Vec<double, 8> > get3dFlow(Mat& frame, Mat& prev) {
          Mat_<Point2f> flow = getFlow(frame, prev);

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

          std::vector< Vec<double, 8> > flow_3d;
          for (i = 0; i < flow_size.height; i++) {
            for (j = 0; j < flow_size.width; j++) {
              flow_3d.push_back(
                Vec<double, 8>(
                  i, j, vx(i, j), vy(i, j), vz(i, j), curlx(i, j), curly(i, j), curlz(i, j)
                )
              );
            }
          }

          return flow_3d;
        }
      }
    }
  }
}

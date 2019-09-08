#include <opencv2/video/tracking.hpp>

#include "dense_flow.hpp"

#define MAX_FLOW      10

#define K_ADJACENT    0.25
#define K_CORNER      0.125

namespace parallel_cv {
  namespace commands {
    namespace algorithms {
      namespace dense_flow {
        const cv::Matx33d div_vx_kernel(K_CORNER,   0, -K_CORNER,
                                        K_ADJACENT, 0, -K_ADJACENT,
                                        K_CORNER,   0, -K_CORNER);
        const cv::Matx33d div_vy_kernel( K_CORNER,  K_ADJACENT,  K_CORNER,
                                         0,         0,           0,
                                        -K_CORNER, -K_ADJACENT, -K_CORNER);

        const cv::Matx33d curlz_vx_kernel( K_CORNER,  K_ADJACENT,  K_CORNER,
                                           0,         0,           0,
                                          -K_CORNER, -K_ADJACENT, -K_CORNER);
        const cv::Matx33d curlz_vy_kernel(-K_CORNER,   0, K_CORNER,
                                          -K_ADJACENT, 0, K_ADJACENT,
                                          -K_CORNER,   0, K_CORNER);

        const cv::Matx13d curly_vz_kernel(0.5, 0, -0.5);
        const cv::Matx31d curlx_vz_kernel(-0.5,
                                           0,
                                           0.5);

        inline cv::Mat_<cv::Point2f> getFlow(cv::Mat& frame, cv::Mat& prev) {
          cv::Mat_<cv::Point2f> flow;

          cv::Ptr<cv::DenseOpticalFlow> algorithm = cv::DISOpticalFlow::create(
            cv::DISOpticalFlow::PRESET_MEDIUM
          );
          algorithm->calc(prev, frame, flow);

          return flow;
        }

        std::vector< cv::Vec<double, 8> > get3dFlow(cv::Mat& frame, cv::Mat& prev) {
          cv::Mat_<cv::Point2f> flow = getFlow(frame, prev);

          int i, j;
          cv::Point2f p;
          cv::Size flow_size = flow.size();
          cv::Mat_<double> vx(flow_size, 0.0), vy(flow_size, 0.0), vz(flow_size, 0.0),
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

          cv::filter2D(vx, temp_x, -1, div_vx_kernel);
          cv::filter2D(vy, temp_y, -1, div_vy_kernel);
          vz = temp_x + temp_y;

          cv::filter2D(vx, temp_x, -1, curlz_vx_kernel);
          cv::filter2D(vy, temp_y, -1, curlz_vy_kernel);
          curlz = temp_x + temp_y;

          cv::filter2D(vz, curly, -1, curly_vz_kernel);
          cv::filter2D(vz, curlx, -1, curlx_vz_kernel);

          std::vector< cv::Vec<double, 8> > flow_3d;
          for (i = 0; i < flow_size.height; i++) {
            for (j = 0; j < flow_size.width; j++) {
              flow_3d.push_back(
                cv::Vec<double, 8>(
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

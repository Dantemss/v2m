#include <algorithm>

#include "opencv2/video/tracking.hpp"

#include "sparse_flow.hpp"

using namespace std;
using namespace cv;

#define MAX_CORNERS       1000
#define QUALITY_LEVEL     0.01
#define MIN_DISTANCE      1
#define MASK              noArray()
#define BLOCK_SIZE        3
#define USE_HARRIS        0
#define K                 0.04

#define MAX_LEVEL         3
#define FLAGS             0
#define MIN_EIG_THRESHOLD 0.001

#define MAX_FLOW          10

namespace parallel_cv {
  namespace commands {
    namespace algorithms {
      namespace sparse_flow {
        const Size winSize(10, 10);
        const Size zeroZone(-1, -1);
        const TermCriteria criteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);

        struct Zorder {
          bool operator()(const Vec<double, 8>& a, const Vec<double, 8>& b) {
            size_t dim, msb_dim = 0;
            int val, msb_val = 0;

            for (dim = 0; dim < 2; dim++) {
              val = (int)a[dim] ^ (int)b[dim];

              if (msb_val < val && msb_val < (msb_val ^ val)) {
                msb_dim = dim;
                msb_val = val;
              }
            }

            return a[msb_dim] < b[msb_dim];
          }
        } zorder;

        vector< Vec<double, 8> > getSparseFlow(Mat& frame, Mat& prev) {
          vector<Point2f> prev_features, features;

          goodFeaturesToTrack(prev, prev_features, MAX_CORNERS, QUALITY_LEVEL,
                              MIN_DISTANCE, MASK, BLOCK_SIZE, USE_HARRIS, K);

          vector< Vec<double, 8> > sparse_flow;
          if (prev_features.empty()) return sparse_flow;

          cornerSubPix(prev, prev_features, winSize, zeroZone, criteria);

          vector<uchar> status;
          vector<float> err;
          calcOpticalFlowPyrLK(prev, frame, prev_features, features, status, err, winSize,
                               MAX_LEVEL, criteria, FLAGS, MIN_EIG_THRESHOLD);

          size_t i;
          Size frame_size = frame.size();
          double vx, vy;
          for (i = 0; i < features.size(); i++) {
            if (!status[i] ||
                features[i].x < 0 || features[i].x > frame_size.width ||
                features[i].y < 0 || features[i].y > frame_size.height) continue;

            vx = features[i].x - prev_features[i].x;
            vy = features[i].y - prev_features[i].y;

            if (abs(vx) > MAX_FLOW || abs(vy) > MAX_FLOW) continue;

            sparse_flow.push_back(
              Vec<double, 8>(
                features[i].y, features[i].x, vx, vy, 0, 0, 0, 0
              )
            );
          }

          return sparse_flow;
        } 

        vector< Vec<double, 8> > get3dFlow(Mat& frame, Mat& prev) {
          vector< Vec<double, 8> > flow_3d = getSparseFlow(frame, prev);

          sort(flow_3d.begin(), flow_3d.end(), zorder);

          int flow_size, i, j, k, l, m, n;
          double vz, curlx, curly, curlz;
          flow_size = flow_3d.size();
          for (k = 0; k < flow_size; k++) {
            i = flow_3d[k][0];
            j = flow_3d[k][1];

            vz = 0;
            curlz = 0;
            curly = 0;
            curlx = 0;

            for (l = max(k - 2, 0); l <= min(k + 2, flow_size); l++) {
              m = flow_3d[k][0];
              n = flow_3d[k][1];
            }

            flow_3d[k][4] = vz;
            flow_3d[k][5] = curlx;
            flow_3d[k][6] = curly;
            flow_3d[k][7] = curlz;
          }

          return flow_3d;
        }
      }
    }
  }
}

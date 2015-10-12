#include <cstdio>
#include <iostream>
#include <queue>
#include <ctime>
#include <cerrno>
#include <pthread.h>
#include <semaphore.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define NUM_WORKER_THREADS 8
#define WORK_FUNCTION denseFlow
#define WINDOW_NAME "Dense Flow"
#define VIDEO_PATH "videos/video.mpeg"

class Workable {
  Mat (&f)(Mat&, Mat&);
  Mat fr, pr, out;
  bool worked;
  pthread_mutex_t mutex;

  public:

  Workable(Mat (&func)(Mat&, Mat&), const Mat& frame, const Mat& prev): f(func) {
    frame.copyTo(fr);
    prev.copyTo(pr);
    worked = false;
    pthread_mutex_init(&mutex, NULL);
  }

  virtual ~Workable() {
    pthread_mutex_destroy(&mutex);
  }

  void work() {
    pthread_mutex_lock(&mutex);
    if (worked) return;
    out = f(fr, pr);
    worked = true;
    pthread_mutex_unlock(&mutex);
  }

  Mat getOutput() {
    if (!worked) work();

    return out;
  }
};

class WorkStream {
  queue < Ptr<Workable> > q;
  pthread_mutex_t mutex;
  sem_t semaphore;

  public:

  WorkStream() {
    pthread_mutex_init(&mutex, NULL);
    sem_init(&semaphore, 0, 0);
  }

  virtual ~WorkStream() {
    sem_destroy(&semaphore);
    pthread_mutex_destroy(&mutex);
  }

  void push(Ptr<Workable> ptr) {
    pthread_mutex_lock(&mutex);
    q.push(ptr);
    pthread_mutex_unlock(&mutex);

    sem_post(&semaphore);
  }

  Ptr<Workable> pop() {
    Ptr<Workable> ptr;

    sem_wait(&semaphore);

    pthread_mutex_lock(&mutex);
    ptr = q.front();
    q.pop();
    pthread_mutex_unlock(&mutex);

    return ptr;
  }
};

Mat workExit(Mat& frame, Mat& prev) {
  pthread_exit(NULL);
  return frame;
}

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1)
{
    dst = Mat::zeros(flow.size(), CV_8UC1);

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                Point2f u = flow(y, x);

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f u = flow(y, x);

            if (isFlowCorrect(u) && (u.x > 0.1 || u.y > 0.1))
                dst.at<uchar>(y, x) = 1;
        }
    }
}

/**
 * @function DenseFlow
 * @brief Calculate the dense optical flow and display it
 */
Mat denseFlow(Mat& frame, Mat& prev) {
  if (prev.empty()) return frame;

  Mat gray, prev_gray;
  cvtColor( frame, gray, CV_BGR2GRAY );
  cvtColor( prev, prev_gray, CV_BGR2GRAY );

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

class WorkParameters {
  WorkStream* stream;
  double mspf;

  public:

  WorkParameters(WorkStream& work_stream, double milliseconds_per_frame) {
    stream = &work_stream;
    mspf = milliseconds_per_frame;
  }

  WorkStream* getStream() {
    return stream;
  }

  double getMspf() {
    return mspf;
  }
};

void *work(void *arg) {
  WorkParameters* params = (WorkParameters*) arg;

  for (;;) params->getStream()->pop()->work();
}

void *output(void *arg) {
  WorkParameters* params = (WorkParameters*) arg;

  for (;;) {
    imshow(WINDOW_NAME, params->getStream()->pop()->getOutput());

    if ((char)waitKey(params->getMspf()) == 27) {
      break;
    }
  }
}

/**
 * @function main
 */
int main(void) {
  VideoCapture capture(VIDEO_PATH);

  if (!capture.isOpened()) {
    printf("Error opening video capture!\n");
    return -1;
  }

  double fps = capture.get(CV_CAP_PROP_FPS);
  double mspf = 1000/fps;

  WorkStream work_stream;
  WorkStream output_stream;

  WorkParameters work_params(work_stream, mspf);
  WorkParameters output_params(output_stream, mspf);

  stringstream title;
  title << WINDOW_NAME << " - " << fps << " FPS";
  setWindowTitle(WINDOW_NAME, title.str());

  int i;
  pthread_t worker_threads[NUM_WORKER_THREADS];
  for (i = 0; i < NUM_WORKER_THREADS; i++) {
    if (pthread_create(&worker_threads[i], NULL, &work, &work_params)) exit(1);
    pthread_detach(worker_threads[i]);
  }

  pthread_t output_thread;
  if (pthread_create(&output_thread, NULL, &output, &output_params)) exit(1);

  timespec tspec;
  Mat frame, prev;
  Ptr<Workable> workable_ptr;
  for (;;) {
    clock_gettime(CLOCK_MONOTONIC, &tspec);

    if (!capture.read(frame) || frame.empty()) break;

    workable_ptr = makePtr<Workable>(WORK_FUNCTION, frame, prev);

    work_stream.push(workable_ptr);
    output_stream.push(workable_ptr);

    cv::swap(frame, prev);

    tspec.tv_nsec += mspf*1e6;
    if (tspec.tv_nsec >= 1e9) {
      tspec.tv_nsec -= 1e9;
      tspec.tv_sec++;
    }
    while(clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &tspec, NULL) == EINTR);
  }

  for (i = 0; i < NUM_WORKER_THREADS; i++) {
    workable_ptr = makePtr<Workable>(workExit, frame, prev);
    work_stream.push(workable_ptr);
  }

  workable_ptr = makePtr<Workable>(workExit, frame, prev);
  output_stream.push(workable_ptr);

  pthread_join(output_thread, NULL);

  return 0;
}

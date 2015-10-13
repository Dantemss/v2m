#include <ctime>
#include <cerrno>
#include <iostream>
#include <pthread.h>

#include "opencv2/highgui/highgui.hpp"

#include "parallel_cv.hpp"
#include "work_stream.hpp"

using namespace cv;

#define WINDOW_NAME "ParallelCV"

namespace parallel_cv {
  void run(cv::String video_path,
           int num_worker_threads,
           cv::Mat (*const work_function)(cv::Mat&, cv::Mat&)) {
    VideoCapture capture(video_path);

    if (!capture.isOpened()) {
      printf("Error opening video capture\n");
      ::exit(EXIT_FAILURE);
    }

    double fps = capture.get(CV_CAP_PROP_FPS);

    WorkStream work_stream(fps);
    WorkStream output_stream(fps);

    int i;
    pthread_t worker_threads[num_worker_threads];
    for (i = 0; i < num_worker_threads; i++) {
      if (pthread_create(&worker_threads[i], NULL, &work, &work_stream)) {
        printf("Error creating worker threads\n");
        ::exit(EXIT_FAILURE);
      }
      pthread_detach(worker_threads[i]);
    }

    pthread_t output_thread;
    if (pthread_create(&output_thread, NULL, &output, &output_stream)) {
      printf("Error creating output thread\n");
      ::exit(EXIT_FAILURE);
    }

    timespec tspec;
    Mat frame, prev;
    Ptr<Workable> workable_ptr;
    for (;;) {
      clock_gettime(CLOCK_MONOTONIC, &tspec);

      if (!capture.read(frame) || frame.empty()) break;

      workable_ptr = makePtr<Workable>(work_function, frame, prev);

      work_stream.push(workable_ptr);
      output_stream.push(workable_ptr);

      cv::swap(frame, prev);

      tspec.tv_nsec += 1e9/fps;
      if (tspec.tv_nsec >= 1e9) {
        tspec.tv_nsec -= 1e9;
        tspec.tv_sec++;
      }
      while(clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &tspec, NULL) == EINTR);
    }

    for (i = 0; i < num_worker_threads; i++) {
      workable_ptr = makePtr<Workable>(&parallel_cv::exit, frame, prev);
      work_stream.push(workable_ptr);
    }

    workable_ptr = makePtr<Workable>(&parallel_cv::exit, frame, prev);
    output_stream.push(workable_ptr);

    pthread_join(output_thread, NULL);
  }

  void *work(void *arg) {
    WorkStream* stream = (WorkStream*) arg;

    for (;;) stream->pop()->work();
  }

  void *output(void *arg) {
    WorkStream* stream = (WorkStream*) arg;
    double fps = stream->getFps();
    double mspf = 1000/fps;
    std::stringstream title;
    title << WINDOW_NAME << " - " << fps << " FPS";
    setWindowTitle(WINDOW_NAME, title.str());

    for (;;) {
      imshow(WINDOW_NAME, stream->pop()->getOutput());

      if ((char)waitKey(mspf) == 27) {
        break;
      }
    }
  }

  Mat exit(Mat& frame, Mat& prev) {
    pthread_exit(NULL);
    return frame;
  }
}

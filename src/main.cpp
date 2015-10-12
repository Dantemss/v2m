#include <cstdio>
#include <ctime>
#include <cerrno>
#include <pthread.h>

#include "opencv2/highgui/highgui.hpp"

#include "work_stream.hpp"
#include "workable.hpp"
#include "work.hpp"
#include "dense_flow.hpp"

using namespace cv;

#define VIDEO_PATH "videos/video.mpeg"
#define NUM_WORKER_THREADS 8
#define WORK_FUNCTION denseFlow

int main(void) {
  VideoCapture capture(VIDEO_PATH);

  if (!capture.isOpened()) {
    printf("Error opening video capture!\n");
    return -1;
  }

  double fps = capture.get(CV_CAP_PROP_FPS);

  WorkStream work_stream(fps);
  WorkStream output_stream(fps);

  int i;
  pthread_t worker_threads[NUM_WORKER_THREADS];
  for (i = 0; i < NUM_WORKER_THREADS; i++) {
    if (pthread_create(&worker_threads[i], NULL, &Work::work, &work_stream)) exit(1);
    pthread_detach(worker_threads[i]);
  }

  pthread_t output_thread;
  if (pthread_create(&output_thread, NULL, &Work::output, &output_stream)) exit(1);

  timespec tspec;
  Mat frame, prev;
  Ptr<Workable> workable_ptr;
  for (;;) {
    clock_gettime(CLOCK_MONOTONIC, &tspec);

    if (!capture.read(frame) || frame.empty()) break;

    workable_ptr = makePtr<Workable>(*WORK_FUNCTION, frame, prev);

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

  for (i = 0; i < NUM_WORKER_THREADS; i++) {
    workable_ptr = makePtr<Workable>(&Work::exit, frame, prev);
    work_stream.push(workable_ptr);
  }

  workable_ptr = makePtr<Workable>(&Work::exit, frame, prev);
  output_stream.push(workable_ptr);

  pthread_join(output_thread, NULL);

  return 0;
}

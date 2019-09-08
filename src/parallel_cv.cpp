#include <cstdio>
#include <ctime>
#include <cerrno>

#include <pthread.h>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "parallel_cv.hpp"
#include "parallel_cv/work_stream.hpp"
#include "parallel_cv/worker.hpp"
#include "parallel_cv/commands/exit.hpp"

using namespace cv;

namespace parallel_cv {
  void log(const char* message) {
    printf("[ParallelCV] %s\n", message);
  }

  void run(cv::String video_path,
           size_t num_worker_threads,
           cv::Mat (*const command_function)(cv::Mat&, cv::Mat&)) {
    VideoCapture capture(video_path);

    if (!capture.isOpened()) {
      log("Error: could not open video capture");
      exit(EXIT_FAILURE);
    }

    double fps = capture.get(CAP_PROP_FPS);

    WorkStream work_stream(fps, 1);
    WorkStream output_stream(fps, num_worker_threads);

    size_t i;
    pthread_t worker_threads[num_worker_threads];
    for (i = 0; i < num_worker_threads; i++) {
      if (pthread_create(&worker_threads[i], NULL, &worker::work, &work_stream)) {
        log("Error: could not create worker threads");
        exit(EXIT_FAILURE);
      }
      pthread_detach(worker_threads[i]);
    }

    pthread_t output_thread;
    if (pthread_create(&output_thread, NULL, &worker::output, &output_stream)) {
      log("Error: could not create output thread");
      exit(EXIT_FAILURE);
    }

    timespec tspec;
    Mat frame, prev;
    Ptr<Workable> workable_ptr;
    for (;;) {
      clock_gettime(CLOCK_MONOTONIC, &tspec);

      if (!capture.read(frame) || frame.empty()) break;

      if (!work_stream.full() && !output_stream.full()) {
        workable_ptr = Ptr<Workable>(new Workable(command_function, frame, prev));

        work_stream.push(workable_ptr);
        output_stream.push(workable_ptr);
      }

      swap(frame, prev);

      tspec.tv_nsec += 1e9/fps;
      while (tspec.tv_nsec >= 1e9) {
        tspec.tv_nsec -= 1e9;
        tspec.tv_sec++;
      }
      while(clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &tspec, NULL) == EINTR);
    }

    workable_ptr = Ptr<Workable>(new Workable(&commands::exit, frame, prev));
    for (i = 0; i < num_worker_threads; i++) {
      work_stream.push(workable_ptr);
    }

    output_stream.push(workable_ptr);

    pthread_join(output_thread, NULL);
  }
}

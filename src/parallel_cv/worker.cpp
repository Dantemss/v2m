#include <ctime>
#include <cerrno>

#include <opencv2/highgui/highgui.hpp>

#include "worker.hpp"
#include "log.hpp"

#include "commands/map.hpp"
#include "commands/exit.hpp"

#define COMMAND_FUNCTION commands::map
#define WINDOW_NAME "ParallelCV"

namespace parallel_cv {
  Worker::Worker(cv::String video_path): capture(video_path), fps(capture.get(cv::CAP_PROP_FPS)) {
    if (!capture.isOpened()) {
      log("Error: could not open video capture");
      exit(EXIT_FAILURE);
    }
  }

  void Worker::begin() {
    if (pthread_create(&input_thread, NULL, &input, this)) {
      log("Error: could not create input thread");
      exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NUM_WORKER_THREADS; i++) {
      if (pthread_create(&worker_threads[i], NULL, &work, this)) {
        log("Error: could not create worker thread");
        exit(EXIT_FAILURE);
      }
    }

    if (pthread_create(&output_thread, NULL, &output, this)) {
      log("Error: could not create output thread");
      exit(EXIT_FAILURE);
    }
  }

  void Worker::start() {
    begin();

    if (pthread_detach(input_thread)) {
      log("Error: could not detach input thread");
      exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NUM_WORKER_THREADS; i++) {
      if (pthread_detach(worker_threads[i])) {
        log("Error: could not detach worker thread");
        exit(EXIT_FAILURE);
      }
    }

    if (pthread_detach(output_thread)) {
      log("Error: could not detach output thread");
      exit(EXIT_FAILURE);
    }
  }

  int Worker::run() {
    begin();

    if (pthread_join(input_thread, NULL)) {
      log("Error: could not join input thread");
      exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NUM_WORKER_THREADS; i++) {
      if (pthread_join(worker_threads[i], NULL)) {
        log("Error: could not join worker thread");
        exit(EXIT_FAILURE);
      }
    }

    if (pthread_join(output_thread, NULL)) {
      log("Error: could not join output thread");
      exit(EXIT_FAILURE);
    }

    return return_value;
  }

  void Worker::stop() {
    stopped = true;
  }

  void* Worker::input(void *arg) {
    Worker* worker = (Worker*) arg;
    cv::Mat prev;

    while (!worker->stopped) {
      timespec time;
      cv::Mat frame;
      cv::Ptr<Workable> workable_ptr;

      clock_gettime(CLOCK_MONOTONIC, &time);

      if (!worker->capture.read(frame) || frame.empty()) break;

      worker->count++;

      if (!worker->work_stream.full() && !worker->output_stream.full()) {
        workable_ptr = cv::Ptr<Workable>(
          new Workable(COMMAND_FUNCTION, frame, prev, worker->count)
        );

        prev.release();

        worker->work_stream.push(workable_ptr);
        worker->output_stream.push(workable_ptr);
      }

      cv::swap(frame, prev);

      time.tv_nsec += 1e9/worker->fps;
      while (time.tv_nsec >= 1e9) {
        time.tv_nsec -= 1e9;
        time.tv_sec++;
      }
      while(clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &time, NULL) == EINTR);
    }

    worker->finished = true;

    return NULL;
  }

  void* Worker::work(void *arg) {
    Worker* worker = (Worker*) arg;

    while (!worker->stopped && (!worker->finished || worker->work_stream.size() > 0)) {
      cv::Ptr<Workable> workable = worker->work_stream.pop();
      if (workable) workable->work();
    }

    return NULL;
  }

  void* Worker::output(void *arg) {
    Worker* worker = (Worker*) arg;
    double mspf = 1000/worker->fps;
    int count = 0;

    while (
      !worker->stopped && (
        !worker->finished || worker->output_stream.size() > 0 || worker->work_stream.size() > 0
      )
    ) {
      cv::Ptr<Workable> workable = worker->output_stream.pop();
      if (workable) {
        count++;
        std::stringstream title;
        title << WINDOW_NAME " - Input FPS: " << worker->fps << " - Input: " << worker->count
              << " - Output: " << count << " - Skipped: " << workable->count - count;
        cv::setWindowTitle(WINDOW_NAME, title.str());
        imshow(WINDOW_NAME, workable->getOutput());

        if ((char) cv::waitKey(mspf) == 27) {
          worker->stop();

          return NULL;
        }
      }
    }

    worker->return_value = 0;

    return NULL;
  }
}

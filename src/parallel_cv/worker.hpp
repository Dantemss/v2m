#ifndef PARALLEL_CV_WORKER_HPP
#define PARALLEL_CV_WORKER_HPP

#include <pthread.h>

#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

#include "work_stream.hpp"

#define NUM_WORKER_THREADS 4

namespace parallel_cv {
  class Worker {
    cv::VideoCapture capture;
    double fps;
    WorkStream work_stream { 1 };
    WorkStream output_stream { NUM_WORKER_THREADS };

    pthread_t input_thread, worker_threads[NUM_WORKER_THREADS], output_thread;

    std::atomic<size_t> count { 0 };
    std::atomic<bool> finished { false };
    std::atomic<bool> stopped { false };
    int return_value { 1 };

    void begin();

    static void* input(void* arg);

    static void* work(void* arg);

    static void* output(void* arg);

    public:

    Worker(cv::String);

    void start();

    int run();

    void stop();
  };
}

#endif

#ifndef V2M_WORKABLE_HPP
#define V2M_WORKABLE_HPP

#include <atomic>

#include <pthread.h>

#include <opencv2/core/mat.hpp>

namespace v2m {
  class Workable {
    cv::Mat (*function)(cv::Mat&, cv::Mat&);
    cv::Mat frame, prev, output;
    std::atomic<bool> worked { false };
    pthread_mutex_t mutex;

    public:

    size_t count;

    Workable(cv::Mat (*const)(cv::Mat&, cv::Mat&), const cv::Mat&, const cv::Mat&, size_t);

    virtual ~Workable();

    void work();

    cv::Mat getOutput();
  };
}

#endif

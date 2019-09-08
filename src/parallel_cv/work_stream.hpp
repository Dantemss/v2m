#ifndef PARALLEL_CV_WORK_STREAM_HPP
#define PARALLEL_CV_WORK_STREAM_HPP

#include <queue>
#include <pthread.h>
#include <semaphore.h>

#include "workable.hpp"

namespace parallel_cv {
  class WorkStream {
    double fps;
    size_t ms;
    std::queue < cv::Ptr<Workable> > q;
    pthread_mutex_t mutex;
    sem_t semaphore;

    public:

    WorkStream(double frames_per_second, size_t max_size);

    virtual ~WorkStream();

    void push(cv::Ptr<Workable> ptr);

    cv::Ptr<Workable> pop();

    bool full();

    double getFps();
  };
}

#endif

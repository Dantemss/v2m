#ifndef V2M_WORK_STREAM_HPP
#define V2M_WORK_STREAM_HPP

#include <queue>
#include <pthread.h>
#include <semaphore.h>

#include "workable.hpp"

namespace v2m {
  class WorkStream {
    size_t max_size;
    std::queue <cv::Ptr<Workable>> queue;
    pthread_mutex_t mutex;
    sem_t semaphore;

    public:

    WorkStream(size_t);

    virtual ~WorkStream();

    void push(cv::Ptr<Workable>);

    cv::Ptr<Workable> pop();

    size_t size();

    bool full();
  };
}

#endif

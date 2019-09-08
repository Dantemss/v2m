#include "work_stream.hpp"

using namespace cv;
using namespace parallel_cv;

WorkStream::WorkStream(double frames_per_second, size_t max_size) {
  fps = frames_per_second;
  ms = max_size;
  pthread_mutex_init(&mutex, NULL);
  sem_init(&semaphore, 0, 0);
}

WorkStream::~WorkStream() {
  sem_destroy(&semaphore);
  pthread_mutex_destroy(&mutex);
}

void WorkStream::push(Ptr<Workable> ptr) {
  pthread_mutex_lock(&mutex);
  q.push(ptr);
  pthread_mutex_unlock(&mutex);

  sem_post(&semaphore);
}

Ptr<Workable> WorkStream::pop() {
  Ptr<Workable> ptr;

  sem_wait(&semaphore);

  pthread_mutex_lock(&mutex);
  ptr = q.front();
  q.pop();
  pthread_mutex_unlock(&mutex);

  return ptr;
}

bool WorkStream::full() {
  return q.size() >= ms;
}

double WorkStream::getFps() {
  return fps;
}

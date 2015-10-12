#include "work_stream.hpp"

using namespace cv;

WorkStream::WorkStream(double frames_per_second) {
  fps = frames_per_second;
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

double WorkStream::getFps() {
  return fps;
}

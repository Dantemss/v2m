#include "work_stream.hpp"

using namespace v2m;

WorkStream::WorkStream(size_t _max_size): max_size(_max_size) {
  pthread_mutex_init(&mutex, NULL);
  sem_init(&semaphore, 0, 0);
}

WorkStream::~WorkStream() {
  sem_destroy(&semaphore);
  pthread_mutex_destroy(&mutex);
}

void WorkStream::push(cv::Ptr<Workable> ptr) {
  pthread_mutex_lock(&mutex);
  queue.push(ptr);
  pthread_mutex_unlock(&mutex);

  sem_post(&semaphore);
}

cv::Ptr<Workable> WorkStream::pop() {
  cv::Ptr<Workable> ptr;

  if (!sem_trywait(&semaphore)) {
    pthread_mutex_lock(&mutex);
    ptr = queue.front();
    queue.pop();
    pthread_mutex_unlock(&mutex);
  }

  return ptr;
}

size_t WorkStream::size() {
  return queue.size();
}

bool WorkStream::full() {
  pthread_mutex_lock(&mutex);
  bool result = size() >= max_size;
  pthread_mutex_unlock(&mutex);

  return result;
}

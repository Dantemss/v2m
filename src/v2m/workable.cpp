#include <unistd.h>

#include "workable.hpp"

using namespace v2m;

Workable::Workable(
  cv::Mat (*const _function)(cv::Mat&, cv::Mat&),
  const cv::Mat& _frame,
  const cv::Mat& _prev,
  size_t _count
): function(_function), frame(_frame), prev(_prev), count(_count) {
  pthread_mutex_init(&mutex, NULL);
}

Workable::~Workable() {
  pthread_mutex_destroy(&mutex);
}

void Workable::work() {
  pthread_mutex_lock(&mutex);
  if (worked) return;
  output = (*function)(frame, prev);
  worked = true;
  pthread_mutex_unlock(&mutex);
}

cv::Mat Workable::getOutput() {
  while (!worked) sleep(0);

  return output;
}

#include "workable.hpp"

using namespace cv;
using namespace parallel_cv;

Workable::Workable(Mat (*const func)(Mat&, Mat&), const Mat& frame, const Mat& prev) {
  f = func;
  frame.copyTo(fr);
  prev.copyTo(pr);
  worked = false;
  pthread_mutex_init(&mutex, NULL);
}

Workable::~Workable() {
  pthread_mutex_destroy(&mutex);
}

void Workable::work() {
  pthread_mutex_lock(&mutex);
  if (worked) return;
  out = (*f)(fr, pr);
  worked = true;
  pthread_mutex_unlock(&mutex);
}

Mat Workable::getOutput() {
  if (!worked) work();

  return out;
}

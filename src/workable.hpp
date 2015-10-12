#ifndef WORKABLE_HPP
#define WORKABLE_HPP

#include <pthread.h>

#include "opencv2/core/mat.hpp"

class Workable {
  cv::Mat (*f)(cv::Mat&, cv::Mat&);
  cv::Mat fr, pr, out;
  bool worked;
  pthread_mutex_t mutex;

  public:

  Workable(cv::Mat (*const func)(cv::Mat&, cv::Mat&), const cv::Mat& frame, const cv::Mat& prev);

  virtual ~Workable();

  void work();

  cv::Mat getOutput();
};

#endif

#include <iostream>

#include "opencv2/highgui/highgui.hpp"

#include "worker.hpp"
#include "work_stream.hpp"

using namespace cv;

#define WINDOW_NAME "ParallelCV"

namespace parallel_cv {
  namespace worker {
    void *work(void *arg) {
      WorkStream* stream = (WorkStream*) arg;

      for (;;) stream->pop()->work();

      return 0;
    }

    void *output(void *arg) {
      WorkStream* stream = (WorkStream*) arg;
      double fps = stream->getFps();
      double mspf = 1000/fps;
      std::stringstream title;
      title << WINDOW_NAME " - " << fps << " FPS";
      setWindowTitle(WINDOW_NAME, title.str());

      for (;;) {
        imshow(WINDOW_NAME, stream->pop()->getOutput());

        if ((char)waitKey(mspf) == 27) {
          break;
        }
      }

      return 0;
    }
  }
}

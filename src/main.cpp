#include "v2m/log.hpp"
#include "v2m/worker.hpp"

using namespace v2m;

int main(int argc, char* argv[]) {
  cv::String video_path;

  switch(argc) {
    case 1: video_path = "0";
            break;
    case 2: video_path = argv[1];
            break;
    default: log("Error: too many arguments");
             return 1;
  }

  Worker worker(video_path);

  return worker.run();
}

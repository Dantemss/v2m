#include <cstdio>

#include "log.hpp"

namespace parallel_cv {
  void log(const char* message) {
    printf("[ParallelCV] %s\n", message);
  }
}

#include <cstdio>

#include "log.hpp"

namespace v2m {
  void log(const char* message) {
    printf("[V2M] %s\n", message);
  }
}

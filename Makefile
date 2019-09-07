CC = g++
CC_FLAGS += -pthread -Og -Wall `pkg-config --cflags opencv4`
LD_FLAGS += -pthread `pkg-config --libs opencv4`

PROG = map

SUBDIRS = src src/parallel_cv src/parallel_cv/commands src/parallel_cv/commands/algorithms

CPP_FILES := $(wildcard $(addsuffix /*.cpp, $(SUBDIRS)))
OBJ_FILES := $(CPP_FILES:src/%.cpp=build/%.o)
EXEC_FILE := $(addprefix bin/, $(PROG))

build/%.o: src/%.cpp
	$(CC) -c $(CC_FLAGS) -o $@ $<

$(PROG): $(OBJ_FILES)
	$(CC) $(OBJ_FILES) -o $(EXEC_FILE) $(LD_FLAGS)

.PHONY: all clean

all: $(EXEC_FILE)

clean:
	rm -f $(OBJ_FILES) $(EXEC_FILE)

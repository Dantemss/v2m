CC=g++
CFLAGS+=`pkg-config --cflags opencv`
LFLAGS+=`pkg-config --libs opencv`

PROG=parallel_cv
SRC=main parallel_cv work_stream workable dense_flow

OBJS=$(addprefix build/, $(addsuffix .o, $(SRC)))
EXEC=bin/$(PROG)

build/%.o: src/%.cpp
	$(CC) -pthread -c -o $@ $< $(CFLAGS)

$(PROG): $(OBJS)
	$(CC) -pthread -o $(EXEC) $(OBJS) $(LFLAGS)

.PHONY: all clean

all: $(PROG)

clean:
	rm -f $(OBJS) $(EXEC)

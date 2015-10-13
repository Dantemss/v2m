CC=g++
CFLAGS+=-pthread -Og -Wall `pkg-config --cflags opencv`
LDFLAGS+=-pthread `pkg-config --libs opencv`

PROG=dense_flow
SRC=main parallel_cv parallel_cv/work_stream parallel_cv/workable parallel_cv/worker $(PROG)

OBJS=$(addprefix build/, $(addsuffix .o, $(SRC)))
EXEC=bin/$(PROG)

build/%.o: src/%.cpp
	$(CC) -c $(CFLAGS) -o $@ $<

$(PROG): $(OBJS)
	$(CC) $(OBJS) -o $(EXEC) $(LDFLAGS)

.PHONY: all clean

all: $(PROG)

clean:
	rm -f $(OBJS) $(EXEC)

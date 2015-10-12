CC=g++
CFLAGS+=`pkg-config --cflags opencv`
LFLAGS+=`pkg-config --libs opencv`

PROG=main
SRC=main

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

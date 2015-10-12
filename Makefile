CC=g++
CFLAGS+=`pkg-config --cflags opencv`
LFLAGS+=`pkg-config --libs opencv`

PROG=main
OBJS=build/$(PROG).o
EXEC=bin/$(PROG)

.PHONY: all clean

build/%.o: src/%.cpp
	$(CC) -pthread -c -o $@ $< $(CFLAGS)

$(PROG): $(OBJS)
	$(CC) -pthread -o $(EXEC) $(OBJS) $(LFLAGS)

all: $(PROG)

clean:
	rm -f $(OBJS) $(EXEC)

CC=g++
CFLAGS+=`pkg-config --cflags opencv`
LFLAGS+=`pkg-config --libs opencv`

PROG=main
OBJS=build/$(PROG).o

.PHONY: all clean

build/%.o: src/%.cpp
	$(CC) -pthread -c -o $@ $(CFLAGS) $<

$(PROG): $(OBJS)
	$(CC) -pthread -o bin/$(PROG) $(OBJS) $(LFLAGS)

all: $(PROG)

clean:
	rm -f $(OBJS) bin/$(PROG)

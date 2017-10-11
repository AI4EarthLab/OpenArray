CXX = g++

CC = gcc

OBJS = calcTime.o gettimeofday.o

FILES = calcTime

all: $(FILES)

%.o: %.c
	$(CC) -c $< -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@

calcTime: $(OBJS)
	$(CXX) -o calcTime $(OBJS)

clean:
	rm -f $(OBJS)
	rm -f $(FILES)

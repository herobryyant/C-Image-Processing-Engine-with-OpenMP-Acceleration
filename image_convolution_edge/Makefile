CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -fopenmp
TARGET = edge_detect

all: $(TARGET)

$(TARGET): edge_detect.c
	$(CC) $(CFLAGS) edge_detect.c -o $(TARGET)

clean:
	rm -f $(TARGET) *.pgm

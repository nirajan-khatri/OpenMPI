# Compiler and flags (use gcc/14.3.0 on HPC cluster)
CC = mpicc
CFLAGS = -O2 -Wall -lm

# Targets
all: matmul

matmul: matmul.c
	$(CC) $(CFLAGS) -o matmul matmul.c

clean:
	rm -f matmul

.PHONY: all clean

# Compiler
CC=mpicc

# Binaries to create
BINARIES=hello solver

# Default target creates all binaries
all: $(BINARIES)

# For each binary (depending on bin) compile the source code .c to bin/$@
$(BINARIES): bin 
	$(CC) $@.c -o bin/$@

# Bin creates the directory holding the executables.
bin:
	mkdir bin

# Clean removes the bin folder.
clean:
	rm -rf bin

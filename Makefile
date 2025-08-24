# CC=g++
# NVCC=nvcc -diag-suppress 550,177,20092,549,20011
# CFLAGS=-O3 -g -march=native
# CUDAFLAGS=-arch=sm_80
# LIBFLAGS=-lm
# EXEC=leiden

# all: $(EXEC)

# # Single binary including both CPU and GPU objects
# $(EXEC): main.o leiden.o leiden_cuda.o
# 	$(NVCC) -o $@ main.o leiden.o leiden_cuda.o -Xcompiler="$(CFLAGS)" $(CUDAFLAGS) $(LIBFLAGS)

# # Compile C++ files
# %.o: %.cpp %.h
# 	$(CC) -o $@ -c $< $(CFLAGS)

# # Compile CUDA files
# leiden_cuda.o: leiden.cu leiden.h
# 	$(NVCC) -o $@ -c $< -Xcompiler="$(CFLAGS)" $(CUDAFLAGS)

# clean:
# 	rm -f *.o $(EXEC)

# .PHONY: clean


CC = g++
NVCC = nvcc -diag-suppress 550,177,20092,549,20011
CFLAGS = -O3 -g -march=native
CUDAFLAGS = -arch=sm_80
LIBFLAGS = -lm
EXEC = leiden

all: $(EXEC)

# Final executable
$(EXEC): main.o leiden.o leiden_cuda.o
	$(NVCC) -o $@ main.o leiden.o leiden_cuda.o -Xcompiler="$(CFLAGS)" $(CUDAFLAGS) $(LIBFLAGS)

# Compile C++ files
main.o: main.cpp leiden.h struct.h
	$(CC) -c main.cpp -o main.o $(CFLAGS)

leiden.o: leiden.cpp leiden.h struct.h
	$(CC) -c leiden.cpp -o leiden.o $(CFLAGS)

# Compile CUDA file
leiden_cuda.o: leiden.cu leiden.h struct.h
	$(NVCC) -c leiden.cu -o leiden_cuda.o -Xcompiler="$(CFLAGS)" $(CUDAFLAGS)

clean:
	rm -f *.o $(EXEC)

.PHONY: clean


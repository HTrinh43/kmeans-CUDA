# C++ compiler settings
CC = g++
CPP_SRCS = $(wildcard ./src/*.cpp)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
INC = ./src/
OPTS = -std=c++17 -Wall -Werror -lpthread -O3

# CUDA compiler settings
NVCC = nvcc
CU_SRCS = $(wildcard ./src/*.cu)
CU_OBJS = $(CU_SRCS:.cu=.o)
NVCCFLAGS = -arch=sm_50  # adjust this for your GPU architecture
NVCCFLAGS += --expt-extended-lambda

# Target settings
EXEC = bin/kmeans

all: clean compile

compile: $(CPP_OBJS) $(CU_OBJS)
	$(CC) $^ $(OPTS) -L/usr/local/cuda/lib64 -lcudart -o $(EXEC)

%.o: %.cpp
	$(CC) $(OPTS) -I$(INC) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $< -o $@

clean:
	rm -f $(EXEC) $(CPP_OBJS) $(CU_OBJS)

.PHONY: all clean compile

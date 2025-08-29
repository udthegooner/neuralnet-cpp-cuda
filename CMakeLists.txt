cmake_minimum_required(VERSION 3.15)
project(MNIST_NN_CPU)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include/cpu)

# Source files
file(GLOB CPU_SOURCES "src/cpu/*.cpp")
set(MAIN_SRC "examples/train_cpu.cpp")

# Executable
add_executable(train_cpu ${MAIN_SRC} ${CPU_SOURCES})

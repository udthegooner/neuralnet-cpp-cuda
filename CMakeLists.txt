cmake_minimum_required(VERSION 3.18)
project(MNIST_NN)

option(USE_GPU "Build with CUDA/GPU support" OFF)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Handle GPU Setup if the option is ON
if(USE_GPU)
    # Enable CUDA as a language
    enable_language(CUDA)
    
    # Standard settings for CUDA
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find the CUDA Toolkit (for Thrust and other libraries)
    find_package(CUDAToolkit REQUIRED)
    
    # Define a GPU Preprocessor Macro
    add_definitions(-DUSE_GPU)
    
    # Source files for GPU
    file(GLOB FILE_SOURCES "src/gpu/*.cu")
    set(MAIN_SRC "examples/train_gpu.cu")
    set(TARGET_NAME train_gpu)
    
    message(STATUS "Build Mode: GPU (CUDA)")
else()
    # Source files for CPU
    file(GLOB FILE_SOURCES "src/cpu/*.cpp")
    set(MAIN_SRC "examples/train_cpu.cpp")
    set(TARGET_NAME train_cpu)

    message(STATUS "Build Mode: CPU Only")
endif()

# Include directories
include_directories(include/cpu include/gpu)

# Executable
add_executable(${TARGET_NAME} ${MAIN_SRC} ${FILE_SOURCES})

# Link CUDA libraries if needed
if(USE_GPU)
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)
endif()

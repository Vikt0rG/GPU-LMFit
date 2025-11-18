
cmake_minimum_required(VERSION 3.10)
project(GPU-LMFit LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/make
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cpu
    ${CMAKE_SOURCE_DIR}/include/gpu
)

# Build core library
add_subdirectory(src/cpu)
add_subdirectory(src/gpu)

# Bindings
add_subdirectory(src/binding)

# Examples
# Include the CPU examples directory which defines a helper function
# `add_cpu_example()` and registers any example executables placed under
# `examples/CPU`.
add_subdirectory(examples/CPU)

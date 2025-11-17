
cmake_minimum_required(VERSION 3.10)
project(GPU-LMFit CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project include directories
# ensure the top-level `include` and the `include/cpu` folders are on the compiler include path
include_directories(${CMAKE_SOURCE_DIR}/make ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/cpu)

# Build CPU core library
add_subdirectory(src/cpu)

# Bindings
add_subdirectory(src/binding)

# Examples
# Include the CPU examples directory which defines a helper function
# `add_cpu_example()` and registers any example executables placed under
# `examples/CPU`.
add_subdirectory(examples/CPU)

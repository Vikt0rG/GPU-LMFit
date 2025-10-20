
cmake_minimum_required(VERSION 3.5)
project(GPU-LMFit CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Build CPU core library
add_subdirectory(src/cpu)

# Example: Gauss_1D executable (links to lmfit_cpu library)
add_executable(Gauss_1D
    examples/CPU/Gauss_1D.cpp
)

target_include_directories(Gauss_1D PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(Gauss_1D PRIVATE lmfit_cpu)

set_target_properties(Gauss_1D PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

cmake_minimum_required(VERSION 3.5)

# Example: Gauss_1D
project(Gauss_1D_example CXX)

add_executable(Gauss_1D
    Gauss_1D.cpp
    ${CMAKE_SOURCE_DIR}/include/lm_solver.cpp
)

target_include_directories(Gauss_1D PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(Gauss_1D PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

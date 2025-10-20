cmake_minimum_required(VERSION 3.5)

project(GPU-LMFit CXX)

set(LMFit_CPU_SOURCES
    lm_solver.cpp
)

add_library(lmfit_cpu STATIC ${LMFit_CPU_SOURCES})
target_include_directories(lmfit_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(lmfit_cpu PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

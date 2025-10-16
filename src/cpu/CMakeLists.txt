cmake_minimum_required(VERSION 3.5)

project(myfit_cpu CXX)

set(MYFIT_CPU_SOURCES
    lm_solver.cpp
)

add_library(myfit_cpu STATIC ${MYFIT_CPU_SOURCES})
target_include_directories(myfit_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(myfit_cpu PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

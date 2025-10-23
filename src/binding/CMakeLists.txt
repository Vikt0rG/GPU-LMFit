cmake_minimum_required(VERSION 3.5)

project(GPU-LMFit-bindings CXX)

# Create a module using pybind11 (extern included copy)
# This simple approach uses the header-only pybind11 in extern/pybind11/include

# Add the binding source
set(BINDING_SOURCES
    binding_cpu.cpp
)

add_library(cpu_lmfit MODULE ${BINDING_SOURCES})

target_include_directories(cpu_lmfit PRIVATE
    ${CMAKE_SOURCE_DIR}/extern/pybind11/include
    ${CMAKE_SOURCE_DIR}/include
)

# Link to the CPU library so bindings can call LMFit implementation
target_link_libraries(cpu_lmfit PRIVATE lmfit_cpu)

# Try to find Python includes/libs so Python.h is available
find_package(PythonLibs REQUIRED)
if(PYTHONLIBS_FOUND)
    target_include_directories(cpu_lmfit PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(cpu_lmfit PRIVATE ${PYTHON_LIBRARIES})
endif()

# On Windows, ensure the module file gets the proper suffix
if (WIN32)
    set_target_properties(cpu_lmfit PROPERTIES PREFIX "")
endif()

set_target_properties(cpu_lmfit PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

# Stage the built extension into the build staging area instead of copying
# into the source tree. This keeps build outputs out of source and supports
# packaging from the staging directory (build/staging).
if(NOT DEFINED PY_PACKAGE_DIR)
    set(PY_PACKAGE_DIR "${CMAKE_BINARY_DIR}/staging/package/")
endif()

file(MAKE_DIRECTORY "${PY_PACKAGE_DIR}")

if(WIN32)
    add_custom_command(TARGET cpu_lmfit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cpu_lmfit>
            "${PY_PACKAGE_DIR}/cpu_lmfit.pyd"
        COMMENT "Staging cpu_lmfit built binary to ${PY_PACKAGE_DIR}/cpu_lmfit.pyd"
    )
else()
    # Unix-like: use shared object suffix from the target file
    add_custom_command(TARGET cpu_lmfit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cpu_lmfit>
            "${PY_PACKAGE_DIR}/cpu_lmfit${CMAKE_SHARED_LIBRARY_SUFFIX}"
        COMMENT "Staging cpu_lmfit built binary to ${PY_PACKAGE_DIR}"
    )
endif()

# Register the staged file with CMake's clean mechanism so `make clean` removes it
if(WIN32)
    set(EXT_SUFFIX ".pyd")
else()
    set(EXT_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

set(DEST_PY "${PY_PACKAGE_DIR}/cpu_lmfit${EXT_SUFFIX}")

# Ensure the path is considered by the generator's clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${DEST_PY}")

# Extra: provide an explicit target to clean staged package files
add_custom_target(package_clean
    COMMAND ${CMAKE_COMMAND} -E rm -f "${DEST_PY}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/staging"
    COMMENT "Cleaning staged Python package files at ${PY_PACKAGE_DIR}"
)

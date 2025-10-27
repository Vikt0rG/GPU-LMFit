cmake_minimum_required(VERSION 3.10)

# Include pybind11 from extern. Use the copy in the project root.
add_subdirectory(${CMAKE_SOURCE_DIR}/extern/pybind11 ${CMAKE_BINARY_DIR}/extern/pybind11 EXCLUDE_FROM_ALL)

# Create the python extension module from source
pybind11_add_module(_cpu_lmfit binding_cpu.cpp)

# Link the underlying CPU library into the module
target_link_libraries(_cpu_lmfit PRIVATE lmfit_cpu)

# Prefer static MSVC runtime for the extension if supported
if (MSVC)
	if(POLICY CMP0091)
		set_target_properties(_cpu_lmfit PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	endif()
endif()

# For MinGW/GNU toolchains, prefer static libgcc/libstdc++ to avoid distributing DLLs
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_options(_cpu_lmfit PRIVATE "-static-libgcc" "-static-libstdc++")
endif()

# Optionally, place the built module directly into the python package folder
set_target_properties(_cpu_lmfit PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/package/cpu_lmfit
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/package/cpu_lmfit
)

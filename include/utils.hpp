#pragma once

#include <cstddef>

// Ensure math constants on MSVC and provide a fallback if M_PI is missing for any reason
#ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// Include CUDA host defines if compiling with NVCC/CUDA-aware compiler
#if defined(__CUDACC__)
    #include <crt/host_defines.h>
#endif

// Precision
#ifdef MYGPUFIT_DOUBLE
    #define real double
#else
    #define real float
#endif // MYGPUFIT_DOUBLE

// Define a function type
typedef real (*ModelFuncType)(const real* x, const real* parameterArray);

// Optional derivative function type: fills `out` with partial derivatives
// wrt parameters for the given x and parameter array. The caller provides
// an output buffer of length equal to the model's parameter count.
typedef void (*ModelDerivType)(const real* x, const real* parameterArray, real* out);

// A simple descriptor that groups a model function, optional analytic
// derivative and the number of parameters. If `derivative` is nullptr
// the solver should fallback to numeric (forward) differences.
struct ModelDescriptor {
	std::size_t n_params = 0;
	ModelFuncType func = nullptr;
	ModelDerivType derivative = nullptr;

	ModelDescriptor() = default;
	ModelDescriptor(std::size_t n, ModelFuncType f, ModelDerivType d = nullptr) : n_params(n), func(f), derivative(d) {}
};
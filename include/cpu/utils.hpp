#pragma once

#include <cstddef>

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
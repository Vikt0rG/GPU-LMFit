#pragma once

#include <utils.hpp>


// Templated device-only entry point.
// Model requirements:
//   - __host__ __device__ real model(const real* x, const real* params) const;
//   - optional: __host__ __device__ void derivative(const real* x, const real* params, real* out) const;
// The implementation is in src/gpu/lm_solver.cu and provides a fully device-side
// Levenberg-Marquardt solver. Callers should provide device-callable functors and
// run this from host code; the function will copy inputs to device, run iterations,
// and copy final params back to host.
template <typename Model>
bool gpu_levenberg_marquardt_fit_device(
    size_t n_points,
    size_t n_params,
    const real* x_host,
    const real* y_host,
    real* params_host,
    const Model& model,
    real tol,
    size_t max_iterations,
    real damping
);

// Thin wrapper with the same semantics that forwards to the device driver.
template <typename Model>
bool gpu_levenberg_marquardt_fit(
    size_t n_points,
    size_t n_params,
    const real* x_host,
    const real* y_host,
    real* params_host,
    const Model& model,
    real tol,
    size_t max_iterations,
    real damping
);
#pragma once

// NOTE: Should be the same as lm_solver.hpp but for CUDA kernels.
// Rename to lm_solver.cuh and move to include/
// Here go the declarations of kernels from kernels.cu

#include <cuda_runtime.h>
#include "utils.hpp"

// In example:
// Kernel to compute residuals
__global__ void calculate_residuals(const real* y, const real* f, real* r, size_t n_points);

#pragma once

// NOTE: WIP
// Here go the declarations of kernels from kernels.cu

#include <cuda_runtime.h>
#include "types.hpp"

// In example:
// Kernel to compute residuals
__global__ void calculate_residuals(const real* y, const real* f, real* r, size_t n_points);

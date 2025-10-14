#pragma once

// NOTE: WIP
// Here go all the inline model function declarations and definitions

#include "types.hpp"

// Example model function
__device__ inline real gaussian_model(real const x, const real* p) {
    real A = p[0];
    real mu = p[1];
    real sigma = p[2];
    return A * expf(-0.5f * ((x - mu) / sigma) * ((x - mu) / sigma));
}

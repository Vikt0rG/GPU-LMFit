#pragma once

#include "types.hpp"
#include <cmath>


inline real gaussian_model(real const x, const real* p) {
    real A = p[0];
    real mu = p[1];
    real sigma = p[2];
    return A * expf(-0.5f * ((x - mu) / sigma) * ((x - mu) / sigma));
}
#pragma once

// NOTE: Remove estimator and make similar to ModelDescriptor,
// so that user can define custom estimators in a similar way as custom models.
// Here go all the inline estimator function declarations and definitions

#include "utils.hpp"

// Example least squares estimator
__device__ inline real least_squares_estimator(const real* residuals, size_t n_points) {
    // Actual implementation
}
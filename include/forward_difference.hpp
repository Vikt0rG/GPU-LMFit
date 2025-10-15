#pragma once

#include "types.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>

// This header is used to define the forward difference method for MyGpuFit.


// Templates for epsilon values and step sizes
// Define epsilon values and sqrt function
template<typename T>
struct constants {
    static constexpr T epsilon() { return T(1e-6); }  // safe default
    static constexpr T sqrt_epsilon() { return std::sqrt(epsilon()); }
};

// Float specialization
template<>
struct constants<float> {
    static constexpr float epsilon() { return 1.192092896e-07F; }  // float epsilon
    static constexpr float sqrt_epsilon() { 
        return 3.4526698300124393e-04F;  // sqrt(1.192092896e-07)
    }
};

// Double specialization  
template<>
struct constants<double> {
    static constexpr double epsilon() { return 2.2204460492503131e-16; }  // double epsilon
    static constexpr double sqrt_epsilon() {
        return 1.4901161193847656e-08;  // sqrt(2.2204460492503131e-16)
    }
};

template<typename T>
constexpr T get_step_size() {
    return constants<T>::sqrt_epsilon();
}


// Forward difference gradient calculation
// This function is header-only and marked inline so it can be included
// in multiple translation units without violating the ODR.
// Ownership contract:
// - caller must allocate `perturbedParameters` (array of length numParameters)
// - caller must allocate `gradientArray` (array of length numParameters)
// - this function will NOT free any caller-owned memory
// - uses raw pointers intentionally for later device-porting
// TODO: Tune the step size for better accuracy and stability.
inline void get_forward_difference(
    ModelFuncType modelFunc,
    real const modelValue, // Use this to avoid redundant double calculations
    real const* x,
    real const* parameterArray,
    real* perturbedParameters,
    std::size_t const numParameters, // Need this since we provide a raw pointer
    real* gradientArray) {

    // Clone parameter array for perturbation
    for (std::size_t i = 0; i < numParameters; ++i) {
        perturbedParameters[i] = parameterArray[i];
    }

    // Get gradient by perturbing each parameter
    for (std::size_t i = 0; i < numParameters; ++i) {
        real step_size = get_step_size<real>() * std::max(static_cast<real>(std::fabs(parameterArray[i])), static_cast<real>(1));

        perturbedParameters[i] = parameterArray[i] + step_size;

        real modelValueForward = modelFunc(x, perturbedParameters);
        gradientArray[i] = (modelValueForward - modelValue) / step_size;
        perturbedParameters[i] = parameterArray[i];  // Reset for next parameter
    }
}
#pragma once
#include "./types.h"
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


// Define a function type
typedef float (*ModelFuncType)(real const x, real const * parameterArray);


// Forward difference gradient calculation
void * getForwardDifference(
    ModelFuncType modelFunc,
    real const modelValue, // Use this to avoid redundant double calculations
    real x, 
    real const* parameterArray,
    size_t const numParameters, // Need this since we provide a raw pointer
    real* gradientArray) {

    // Clone parameter array for perturbation
    real * perturbedParameters = new real[numParameters];
    for (int i = 0; i < numParameters; ++i) {
        perturbedParameters[i] = parameterArray[i];
    }

    // Get gradient by perturbing each parameter
    for (int i = 0; i < numParameters; ++i) {
        real step_size = get_step_size<real>() * std::max<real>(static_cast<real>(fabs(parameterArray[i])), static_cast<real>(1.0));

        perturbedParameters[i] = parameterArray[i] + step_size;

        real modelValueForward = modelFunc(x, perturbedParameters);
        gradientArray[i] = (modelValueForward - modelValue) / step_size;
        perturbedParameters[i] = parameterArray[i];  // Reset for next parameter
    }

    // Clean up
    delete[] perturbedParameters;
}
#pragma once

// Precision
#ifdef GPUFIT_DOUBLE
    #define real double
#else
    #define real float
#endif // GPUFIT_DOUBLE


// Status
#include <stdexcept>

#define CUDA_CHECK_STATUS( cuda_function_call ) \
    if (cudaError_t const status = cuda_function_call) \
    { \
        throw std::runtime_error( cudaGetErrorString( status ) ) ; \
    }

#if (defined(_WIN64) || defined(__x86_64__) || defined(__LP64__)) 
    #define ARCH_64
#endif // (defined(_WIN64) || defined(__x86_64__) || defined(__LP64__))
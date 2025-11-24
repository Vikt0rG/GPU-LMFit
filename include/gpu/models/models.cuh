// Aggregator header for all model definitions
#pragma once

#include <cstddef>

#include <gaussian_1d.cuh>


// X-macro list: add a line here when adding a new model
#define MODEL_LIST \
    X(Gaussian1D, gaussian_1d.cuh, GaussianModel)


namespace lmfit {
    enum class ModelId : int {
    #define X(id, header, type) id,
        MODEL_LIST
    #undef X
        Count
};
} // namespace lmfit
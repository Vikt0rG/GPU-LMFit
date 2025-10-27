#pragma once

#include "types.hpp"
#include <cmath>

struct GaussianModel {
    using value_type = real;
    static constexpr int n_params = 3;

    // Evaluate model at *x with parameters p[0]=A, p[1]=mu, p[2]=sigma
    inline value_type model(value_type const* x, value_type const* p) const noexcept {
        value_type A = p[0];
        value_type mu = p[1];
        value_type sigma = p[2];
        value_type z = ((*x) - mu) / sigma;
        return A * static_cast<value_type>(std::exp(-0.5 * z * z));
    }

    // Pointer-based convenience call (same as the old free function signature)
    inline value_type operator()(value_type const* x, value_type const* p) const noexcept {
        return model(x, p);
    }

    // Fill out[0..2] with partial derivatives wrt [A, mu, sigma]
    inline void derivative(value_type const* x, value_type const* p, value_type* out) const noexcept {
        value_type A = p[0];
        value_type mu = p[1];
        value_type sigma = p[2];
        value_type z = ((*x) - mu) / sigma;
        value_type e = static_cast<value_type>(std::exp(-0.5 * z * z));
        out[0] = e;                     // d/dA
        out[1] = A * e * (z / sigma);   // d/dmu
        out[2] = A * e * (z * z / sigma); // d/dsigma
    }
};

// C-compatible wrapper preserving the old free-function API
inline real gaussian_model(const real* x, const real* p) {
    return GaussianModel().model(x, p);
}
#pragma once

#include <utils.hpp>

struct GaussianModel {
    using value_type = real;
    static constexpr int n_params = 3;

    __host__ __device__
    inline value_type model(value_type const* x, value_type const* p) const noexcept {
        value_type A = p[0], mu = p[1], sigma = p[2];
        value_type z = ((*x) - mu) / sigma;
        return A * (value_type) ::exp(-0.5 * z * z);
    }

    __host__ __device__
    inline void derivative(value_type const* x, value_type const* p, value_type* out) const noexcept {
        value_type A = p[0], mu = p[1], sigma = p[2];
        value_type z = ((*x) - mu) / sigma;
        value_type e = (value_type) ::exp(-0.5 * z * z);
        out[0] = e;
        out[1] = A * e * (z / sigma);
        out[2] = A * e * (z * z / sigma);
    }
};
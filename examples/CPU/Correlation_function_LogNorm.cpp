// Example implementation of correlation function for isolated 
// spheres with log-normal size distribution

#include "lm_solver.hpp"
#include <numeric>
#include <vector>
#include <tuple>

real G_term(real z, real d) {
    if (z == 0.0f) {
        return 1.0f; // Edge case: G(0) = 1
    }

    real z_over_2d = z / (2.0f * d);
    real z_over_d = z / d;

    // Term1 calculation
    real term1 = 0.0f;
    if (z < d) {
        real sqrt_term = sqrtf(1.0f - z_over_d * z_over_d);
        term1 = sqrt_term * (1.0f + 0.5f * z_over_d * z_over_d);
    }

    // Term2 calculation
    real term2;
    if (z < d) {
        real prefactor = 2.0f * z_over_d * z_over_d * (1.0f - z_over_d * z_over_d);
        real denominator = 1.0f + sqrtf(1.0f - z_over_d * z_over_d);
        term2 = (denominator > 0.0f) ? prefactor * logf(z_over_d / denominator) : 0.0f;
    }

    real result = term1 + term2;
    return (result > 0.0f) ? result : 0.0f;
}


real lognormal_distribution(real x, real s) {
    real exponent = - 0.5f * (log(x) / s) * (log(x) / s);
    real denominator = s * x * sqrt(2.0f * M_PI);

    return (denominator > 1e-300) ? exp(exponent) / denominator : 0.0f;
}


std::pair<std::vector<real>, std::vector<real>> get_lognormal_distribution(
    real s,
    real scale,
    real loc,
    const int N_LOGNORMAL_POINTS = 1000
) {
    // Generates log-normal distribution points and values
    std::vector<real> distr_range(N_LOGNORMAL_POINTS);
    std::vector<real> distr_values(N_LOGNORMAL_POINTS);

    real mean = log(scale);  // Convert scale back to underlying normal mean
    real stddev = s;         // Shape parameter is the underlying normal stddev

    // Quantiles for the underlying normal distribution
    const real normal_lo = mean - 3.09023f * stddev;  // 0.1% quantile
    const real normal_hi = mean + 3.09023f * stddev;  // 99.9% quantile

    // Linear space in the lognormal variable
    const real lo_lim = loc + exp(normal_lo);
    const real hi_lim = loc + exp(normal_hi);
    const real step = (hi_lim - lo_lim) / (N_LOGNORMAL_POINTS - 1);

    for (size_t i = 0; i < N_LOGNORMAL_POINTS; ++i) {
        distr_range[i] = lo_lim + i * step;

        // Lognormal PDF with location parameter
        real y = (distr_range[i] - loc) / scale;
        distr_values[i] = lognormal_distribution(y, s) / scale;
    }

    return make_pair(distr_range, distr_values);
}


real corr_isol_spheres_polydisp_lognorm(const real* xi, const real* p) {
    real const s = p[0];
    real const scale = p[1];
    real const loc = p[2];

    std::vector<real> distr_range;
    std::vector<real> distr_values;
    std::tie(distr_range, distr_values) = get_lognormal_distribution(s, scale, loc);

    const real sum_probs = std::accumulate(distr_values.begin(), distr_values.end(), 0.0f);

    real result = 0.0f;
    for (size_t i = 0; i < distr_values.size(); ++i) {
        if (distr_values[i] > 0) {
            result += distr_values[i] * G_term(*xi, distr_range[i]);
        }
    }
    result /= sum_probs; // Normalize by total probability

    return result;
}


void create_correlation_data(
    real const* true_params,  // true parameters for data generation
    size_t const n_points,
    real noise_level,
    real* x_data,             // output x data
    real* y_data              // output y data with noise
) {
    real const lower_bound = 0.0f;
    real const upper_bound = 100.0f;
    for (size_t i = 0; i < n_points; ++i) {
        x_data[i] = lower_bound + i * (upper_bound - lower_bound) / (n_points - 1);
        y_data[i] = corr_isol_spheres_polydisp_lognorm(
            &x_data[i],
            true_params
        ) + noise_level * ((rand() % 100) / 100.0f - 0.5f); // add noise
    }
}


bool fit() {
    size_t const n_points = 100;
    size_t const n_params = 3;

    // Simulate correlation data with noise
    real const* true_params = new real[3]{0.2f, 10.0f, 0.0f}; // s, scale, loc
    real* x_data = new real[n_points];
    real* y_data = new real[n_points];
    create_correlation_data(true_params, n_points, 0.01f, x_data, y_data);

    // Initial parameter guesses
    real* initial_params = new real[3]{0.3f, 8.0f, 1.0f}; // initial guesses

    // Create a ModelDescriptor for the correlation function (without analytic derivative)
    ModelDescriptor corr_desc(n_params, &corr_isol_spheres_polydisp_lognorm, nullptr);

    // Fit
    LMFit fitter_new;
    bool success = fitter_new.levenberg_marquardt_fit(
        n_points,
        n_params,
        x_data,
        y_data,
        initial_params,
        corr_desc,
        1e-6,       // tolerance
        10,         // max iterations
        0.001f      // initial damping
    );

    std::cout << "\nNumeric-derivative fit results:\n";
    if (success) fitter_new.print_fit_metrics(); else std::cout << "Numeric fit failed." << std::endl;

    // cleanup
    delete[] initial_params;
    delete[] x_data;
    delete[] y_data;
    delete[] true_params;

    return success;
}


int main() {
    try {
        fit();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
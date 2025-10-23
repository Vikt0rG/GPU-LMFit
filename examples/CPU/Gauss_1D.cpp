// Simplistic 1D Gaussian function fit example

#include "lm_solver.hpp"

// Local Gaussian model implementation and derivative so this example does
// not depend on a separate models.hpp. The model signature matches
// ModelFuncType and the derivative matches ModelDerivType.
inline real gauss_func(const real* x, const real* p) {
    real A = p[0];
    real mu = p[1];
    real sigma = p[2];
    real z = ((*x) - mu) / sigma;
    return A * static_cast<real>(std::exp(-0.5 * z * z));
}


inline void gauss_deriv(const real* x, const real* p, real* out) {
    real A = p[0];
    real mu = p[1];
    real sigma = p[2];
    real z = ((*x) - mu) / sigma;
    real e = static_cast<real>(std::exp(-0.5 * z * z));
    out[0] = e;                    // d/dA
    out[1] = A * e * (z / sigma);  // d/dmu
    out[2] = A * e * (z * z / sigma); // d/dsigma
}


void create_gaussian_data(
    real const* true_params,  // true parameters for data generation
    size_t const n_points,
    real noise_level,
    real* x_data,             // output x data
    real* y_data              // output y data with noise
) {
    real const lower_bound = -5.0;
    real const upper_bound = 5.0;
    for (size_t i = 0; i < n_points; ++i) {
        x_data[i] = lower_bound + i * (upper_bound - lower_bound) / (n_points - 1);
    y_data[i] = gauss_func(&x_data[i], true_params) + noise_level * ((rand() % 100) / 100.0f - 0.5f); // add noise
    }
}


bool fit() {
    size_t const n_points = 100;
    size_t const n_params = 3;

    // Simulate gaussian data with noise
    real const* true_params = new real[3]{6.0, 0.0, 1.0}; // amplitude, mean, stddev
    real* x_data = new real[n_points];
    real* y_data = new real[n_points];
    create_gaussian_data(true_params, n_points, 0.1f, x_data, y_data);

    // Initial parameter guesses
    real* initial_params = new real[3]{3.0, 1.0, 2.0}; // initial guesses

    // Create a ModelDescriptor for the Gaussian (with analytic derivative)
    ModelDescriptor gauss_desc(n_params, gauss_func, gauss_deriv);

    // Fit
    LMFit fitter;
    bool success = fitter.levenberg_marquardt_fit(
        n_points,
        3,
        x_data,
        y_data,
        initial_params,
        gauss_desc,
        1e-6,       // tolerance
        100,        // max iterations
        0.01f       // initial damping
    );
    if (success) fitter.print_fit_metrics(); else std::cout << "Fitting failed." << std::endl;

    // Now run a second fit using numeric forward differences for comparison
    real* initial_params2 = new real[3]{3.0, 1.0, 2.0};
    ModelDescriptor gauss_desc_numeric(n_params, gauss_func, nullptr);
    LMFit fitter2;
    bool success2 = fitter2.levenberg_marquardt_fit(
        n_points,
        3,
        x_data,
        y_data,
        initial_params2,
        gauss_desc_numeric,
        1e-6,
        100,
        0.01f
    );

    std::cout << "\nNumeric-derivative fit results:\n";
    if (success2) fitter2.print_fit_metrics(); else std::cout << "Numeric fit failed." << std::endl;

    delete[] initial_params2;

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
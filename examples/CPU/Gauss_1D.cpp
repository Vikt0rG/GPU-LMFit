// Simplistic 1D Gaussian function fit example

#include <iostream>
#include "models.hpp"
#include "lm_solver.hpp"


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
        y_data[i] = gaussian_model(&x_data[i], true_params) + noise_level * ((rand() % 100) / 100.0f - 0.5f); // add noise
    }
}


void print_metrics(LMFit& fitter) {
    real fitted[3] = {0.0, 0.0, 0.0};
    fitter.copy_optimized_params(fitted, 3);

    // Iterations
    std::cout << "Iterations: " << fitter.get_iterations() << std::endl;

    // Chi^2 history
    std::cout << "Chi^2 history (per iteration):\n";
    const real* history_ptr = fitter.get_chi2_history_ptr();
    std::size_t history_size = fitter.get_chi2_history_size();
    for (size_t i = 0; i < history_size; ++i) {
        std::cout << "  [" << i << "] " << history_ptr[i] << "\n";
    }

    // Summarize fitted parameters
    std::cout << "Fitted params:\n";
    std::cout << "  Amplitude = " << fitted[0] << "\n";
    std::cout << "  Mean      = " << fitted[1] << "\n";
    std::cout << "  StdDev    = " << fitted[2] << "\n";
}


bool fit() {
    size_t const n_points = 100;
    size_t const n_params = 3;

    // Simulate gaussian data with noise
    real const* true_params = new real[3]{5.0, 0.0, 1.0}; // amplitude, mean, stddev
    real* x_data = new real[n_points];
    real* y_data = new real[n_points];
    create_gaussian_data(true_params, n_points, 0.1f, x_data, y_data);

    // Initial parameter guesses
    real* initial_params = new real[3]{4.0, 1.0, 2.0}; // initial guesses

    // Fit
    ModelFuncType model = gaussian_model;

    // Fit
    LMFit fitter;
    bool success = fitter.levenberg_marquardt_fit(
        n_points,
        3,
        x_data,
        y_data,
        initial_params,
        gaussian_model,
        1e-6,       // tolerance
        100,        // max iterations
        0.01f       // initial damping
    );

    if (success) {
        // Print metrics and optimized parameters
        print_metrics(fitter);
    } else {
        std::cout << "Fitting failed." << std::endl;
    }

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
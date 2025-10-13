#include "./types.h"
#include "./utils.h"

// Compute residuals: r = y - f
void compute_residuals(
    int n_points,       // number of data points
    const real* y,      // data points
    const real* f,      // model function values
    real* r             // residuals
) {
    for (int i = 0; i < n_points; ++i) {
        r[i] = y[i] - f[i];
    }
}


// Compute Jacobian matrix J using finite differences
void compute_jacobian(
    int n_points,          // number of data points
    int n_params,          // number of parameters
    const real* x,         // independent variable data points
    const real* params,    // current parameter estimates
    real (*model_func)(real, const real*), // model function pointer (arg1: x, arg2: params)
    real* J                 // Jacobian matrix (n_points x n_params)
) {
    const real h = get_step_size<real>();

    for (int j = 0; j < n_params; ++j) {
        // Create perturbed parameter array
        real* params_perturbed = new real[n_params];
        for (int k = 0; k < n_params; ++k) {
            params_perturbed[k] = params[k];
        }
        params_perturbed[j] += h;

        for (int i = 0; i < n_points; ++i) {
            real f_x = model_func(x[i], params);
            real f_x_perturbed = model_func(x[i], params_perturbed);
            J[i * n_params + j] = (f_x_perturbed - f_x) / h;
        }

        delete[] params_perturbed;
    }
}
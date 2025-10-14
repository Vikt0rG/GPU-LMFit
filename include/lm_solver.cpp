#include "types.hpp"
#include "utils.hpp"
#include "models.hpp"
#include "forward_difference.hpp"


// Compute function values
void compute_function_values(
    size_t n_points,            // number of data points
    const real* x,              // independent variable data points
    const real* params,         // current parameter estimates
    ModelFuncType model_func,   // model function pointer (arg1: x, arg2: ptr to params)
    real* output_values         // model function values
) {
    for (int i = 0; i < n_points; ++i) {
        output_values[i] = model_func(&x[i], params);
    }
}


// Compute residuals: r = y - f
void compute_residuals(
    int n_points,               // number of data points
    const real* y,              // data points
    const real* f,              // model function values
    real* output_r              // residuals
) {
    for (int i = 0; i < n_points; ++i) {
        output_r[i] = y[i] - f[i];
    }
}


// Compute chi-squared
real compute_sum_of_squares(
    size_t n_points,            // number of data points
    const real* r               // residuals
) {
    real sum_squares = 0.0;
    for (size_t i = 0; i < n_points; ++i) {
        sum_squares += r[i] * r[i];
    }
    return sum_squares;
}


// Compute Jacobian using forward difference
// WIP: Optimize by batch processing - compute all gradients at once rather than per parameter
// NOTE: Since host code is there for demonstration purposes, I've just used simple loops here.
// In actual code in ./src use BLAS and device code for GPU acceleration.
void compute_jacobian(
    ModelFuncType model_func,   // model function pointer
    size_t n_points,            // number of data points
    const real* x,              // independent variable data points
    const real* params,         // current parameter estimates
    real* perturbedParams,     // temporary storage for perturbed parameters
    size_t n_params,            // number of parameters
    real* gradient,             // temporary gradient storage
    real* J                     // output Jacobian matrix (row-major: n_points x n_params)
) {
    for (size_t j = 0; j < n_params; ++j) {
        // Compute gradient for parameter j
        getForwardDifference(model_func, model_func(&x[0], params), x, params, perturbedParams, n_params, gradient);

        // Fill in the Jacobian column for parameter j
        for (size_t i = 0; i < n_points; ++i) {
            J[i * n_params + j] = gradient[i];
        }
    }
}


// Cholesky decomposition
bool cholesky_decompose(
    real* A,                    // Input matrix in row-major order
    size_t n                    // Matrix dimension, corresponding to n_params
) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            real sum = A[i * n + j];
            for (size_t k = 0; k < j; ++k) {
                sum -= A[i * n + k] * A[j * n + k];
            }
            if (i == j) {
                if (sum <= 0.0) return false; // Not positive definite
                A[i * n + j] = std::sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0; // Zero out upper triangle
        }
    }
    return true;
}


// Solve Ax = b using Cholesky factors (A = L * L^T)
void cholesky_solve(
    const real* L,              // Lower triangular matrix from Cholesky decomposition
    size_t n,                   // Matrix dimension, corresponding to n_params
    const real* b,              // Right-hand side vector
    real* x                     // Solution vector
) {
    // Solve Ly = b
    real* y = new real[n];
    for (size_t i = 0; i < n; ++i) {
        real sum = b[i];
        for (size_t j = 0; j < i; ++j) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }

    // Solve L^T x = y
    for (size_t i = n; i-- > 0;) {
        real sum = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }

    delete[] y;
}


// Normal equations solver using Cholesky decomposition
// NOTE: Tripple loop can be replaced with optimized BLAS calls for better performance
bool solve_normal_equations(
    size_t n_points,            // number of data points
    size_t n_params,            // number of parameters
    const real* J,              // Jacobian matrix (row-major: n_points x n_params)
    const real* r,              // residuals vector
    real* delta_params,         // output parameter updates
    real damping                // damping factor (lambda)
) {
    real* JTJ = new real[n_params * n_params]();
    real* JTr = new real[n_params]();

    for (size_t i = 0; i < n_params; ++i) {
        for (size_t j = 0; j < n_params; ++j) {
            for (size_t k = 0; k < n_points; ++k) {
                JTJ[i * n_params + j] += J[k * n_params + i] * J[k * n_params + j];
            }
        }
    }

    // Add damping term to diagonal
    for (size_t i = 0; i < n_params; ++i) {
        JTJ[i * n_params + i] += damping;
    }

    for (size_t i = 0; i < n_params; ++i) {
        for (size_t k = 0; k < n_points; ++k) {
            JTr[i] += J[k * n_params + i] * r[k];
        }
    }

    if (!cholesky_decompose(JTJ, n_params)) {
        delete[] JTJ;
        delete[] JTr;
        return false;
    }

    cholesky_solve(JTJ, n_params, JTr, delta_params);

    delete[] JTJ;
    delete[] JTr;
    return true;
}


// Levenberg-Marquardt parameter fitting routine
bool levenberg_marquardt_fit(
    size_t n_points,            // number of data points
    size_t n_params,            // number of parameters
    const real* x,              // independent variable data points
    const real* y,              // data points
    real* params,               // initial parameter estimates (will be updated)
    ModelFuncType model_func,   // model function pointer
    real tol,                   // convergence tolerance
    size_t max_iterations,      // maximum number of iterations
    real damping                // damping factor (lambda)
) {
    // Initializations
    real* function_values = new real[n_points];
    real* gradients = new real[n_points];
    real* function_values_updated = new real[n_points];
    real* residuals = new real[n_points];
    real* residuals_updated = new real[n_points];
    real* perturbedParams = new real[n_params];
    real* jacobian = new real[n_points * n_params];
    real* delta_params = new real[n_params];
    real* params_updated = new real[n_params];
    real chi_squared_current = std::numeric_limits<real>::infinity();

    while (max_iterations-- > 0) {
        // Compute current chi-squared    
        compute_function_values(n_points, x, params, model_func, function_values);
        compute_residuals(n_points, y, function_values, residuals);
        real chi_squared_current = compute_sum_of_squares(n_points, residuals);

        // Solve normal equations to get parameter updates
        compute_jacobian(model_func, n_points, x, params, perturbedParams, n_params, gradients, jacobian);
        if (!solve_normal_equations(n_points, n_params, jacobian, residuals, delta_params, damping)) {
            return false; // Failed to solve
        }

        // Propose new parameters
        for (size_t i = 0; i < n_params; ++i) {
            params_updated[i] = params[i] + delta_params[i];
        }

        // Compute new chi-squared
        compute_function_values(n_points, x, params_updated, model_func, function_values_updated);
        compute_residuals(n_points, y, function_values_updated, residuals_updated);
        real chi_squared_new = compute_sum_of_squares(n_points, residuals_updated);

        // Compute norm of parameter changes for convergence check
        real param_change_norm = 0.0;
        for (size_t i = 0; i < n_params; ++i) {
            param_change_norm += delta_params[i] * delta_params[i];
        }
        param_change_norm = std::sqrt(param_change_norm);

        // Decision making
        if (chi_squared_new < chi_squared_current) {
            // Accept new parameters
            for (size_t i = 0; i < n_params; ++i) {
                params[i] = params_updated[i];
            }
            // Update damping factor
            // TODO: #1 Consider Adaptive Damping based on chi-squared reduction ratio
            damping *= 0.1;
        } else {
            // Reject new parameters, increase damping
            damping *= 10.0;
        }

        // Check for convergence
        if (std::abs(chi_squared_current - chi_squared_new) < tol || param_change_norm < tol) {
            // Converged
            for (size_t i = 0; i < n_params; ++i) {
                params[i] = params_updated[i];
            }
            break;
        }
    }

    // Clean up
    delete[] function_values;
    delete[] gradients;
    delete[] function_values_updated;
    delete[] residuals;
    delete[] residuals_updated;
    delete[] perturbedParams;
    delete[] jacobian;
    delete[] delta_params;
    delete[] params_updated;

    return true;
}
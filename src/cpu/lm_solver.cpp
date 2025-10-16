#include "types.hpp"
#include "utils.hpp"
#include "models.hpp"
#include "forward_difference.hpp"
#include "lm_solver.hpp"
#include <limits>
#include <cmath>


// LMFit constructor / destructor / helpers
LMFit::LMFit() {
}

LMFit::~LMFit() {
    delete[] function_values_;
    delete[] function_values_updated_;
    delete[] gradients_;
    delete[] residuals_;
    delete[] residuals_updated_;
    delete[] perturbed_params_;
    delete[] jacobian_;
    delete[] delta_params_;
    delete[] params_updated_;
    delete[] JTJ_;
    delete[] JTr_;
    delete[] fitted_params_;
    delete[] chi2_history_;
}

void LMFit::ensure_capacity(std::size_t n_points, std::size_t n_params) {
    if (n_points > cap_points_) {
        // free old point-sized buffers
        delete[] function_values_;
        delete[] function_values_updated_;
        delete[] gradients_;
        delete[] residuals_;
        delete[] residuals_updated_;

        cap_points_ = n_points;
        function_values_ = new real[cap_points_];
        function_values_updated_ = new real[cap_points_];
        gradients_ = new real[cap_points_];
        residuals_ = new real[cap_points_];
        residuals_updated_ = new real[cap_points_];
    }
    if (n_params > cap_params_) {
        // free old param-sized buffers
        delete[] perturbed_params_;
        delete[] jacobian_;
        delete[] delta_params_;
        delete[] params_updated_;
        delete[] JTJ_;
        delete[] JTr_;
        delete[] fitted_params_;

        cap_params_ = n_params;
        perturbed_params_ = new real[cap_params_];
        // allocate jacobian with current cap_points_ (may be zero until first ensure_capacity)
        jacobian_ = new real[(cap_points_ > 0 ? cap_points_ : n_points) * cap_params_];
        delta_params_ = new real[cap_params_];
        params_updated_ = new real[cap_params_];
        JTJ_ = new real[cap_params_ * cap_params_];
        JTr_ = new real[cap_params_];
        // allocate storage for fitted params
        fitted_params_ = new real[cap_params_];
    }
}

// Getters for fit metrics
real LMFit::get_chi_squared() const {
    return chi_squared_;
}

std::size_t LMFit::get_iterations() const {
    return iterations_;
}

void LMFit::copy_optimized_params(real* out_params, std::size_t n_params) const {
    if (!out_params) return;
    // copy up to n_params or cap_params_
    std::size_t to_copy = n_params;
    if (to_copy > cap_params_) to_copy = cap_params_;
    for (std::size_t i = 0; i < to_copy; ++i) out_params[i] = fitted_params_[i];
}

const real* LMFit::get_chi2_history_ptr() const {
    return chi2_history_;
}

std::size_t LMFit::get_chi2_history_size() const {
    return chi2_history_size_;
}

// Compute function values
void LMFit::compute_function_values(
    size_t n_points,            // number of data points
    const real* x,              // independent variable data points
    const real* params,         // current parameter estimates
    ModelFuncType model_func,   // model function pointer (arg1: x, arg2: ptr to params)
    real* output_values         // model function values
) {
    for (size_t i = 0; i < n_points; ++i) {
        output_values[i] = model_func(&x[i], params);
    }
}


// Compute residuals: r = y - f
void LMFit::compute_residuals(
    size_t n_points,               // number of data points
    const real* y,              // data points
    const real* f,              // model function values
    real* output_r              // residuals
) {
    for (size_t i = 0; i < n_points; ++i) {
        output_r[i] = y[i] - f[i];
    }
}


// Compute chi-squared
real LMFit::compute_sum_of_squares(
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
void LMFit::compute_jacobian(
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
        get_forward_difference(model_func, model_func(&x[0], params), x, params, perturbedParams, n_params, gradient);

        // Fill in the Jacobian column for parameter j
        for (size_t i = 0; i < n_points; ++i) {
            J[i * n_params + j] = gradient[i];
        }
    }
}

// Cholesky decomposition
// Cholesky decomposition (kept as free helper since it's a low-level op)
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
bool LMFit::solve_normal_equations(
    size_t n_points,            // number of data points
    size_t n_params,            // number of parameters
    const real* J,              // Jacobian matrix (row-major: n_points x n_params)
    const real* r,              // residuals vector
    real* delta_params,         // output parameter updates
    real damping                // damping factor (lambda)
) {
    // Zero JTJ_ and JTr_
    for (size_t idx = 0; idx < n_params * n_params; ++idx) JTJ_[idx] = 0.0;
    for (size_t idx = 0; idx < n_params; ++idx) JTr_[idx] = 0.0;

    for (size_t i = 0; i < n_params; ++i) {
        for (size_t j = 0; j < n_params; ++j) {
            for (size_t k = 0; k < n_points; ++k) {
                JTJ_[i * n_params + j] += J[k * n_params + i] * J[k * n_params + j];
            }
        }
    }

    // Add damping term to diagonal
    for (size_t i = 0; i < n_params; ++i) {
        JTJ_[i * n_params + i] += damping;
    }

    for (size_t i = 0; i < n_params; ++i) {
        for (size_t k = 0; k < n_points; ++k) {
            JTr_[i] += J[k * n_params + i] * r[k];
        }
    }

    if (!cholesky_decompose(JTJ_, n_params)) {
        return false;
    }

    cholesky_solve(JTJ_, n_params, JTr_, delta_params);
    return true;
}

// Levenberg-Marquardt parameter fitting routine
bool LMFit::levenberg_marquardt_fit(
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
    // Ensure buffers are large enough
    ensure_capacity(n_points, n_params);

    chi_squared_ = std::numeric_limits<real>::infinity();
    iterations_ = 0;
    // allocate chi2 history buffer sized by max_iterations if necessary
    if (max_iterations > chi2_history_cap_) {
        delete[] chi2_history_;
        chi2_history_ = new real[max_iterations * 2]; // small growth factor
        chi2_history_cap_ = max_iterations * 2;
    }
    chi2_history_size_ = 0;

    while (max_iterations-- > 0) {
        // count this iteration
        ++iterations_;
        // Compute current chi-squared    
        compute_function_values(n_points, x, params, model_func, function_values_);
        compute_residuals(n_points, y, function_values_, residuals_);
        real chi_squared_current = compute_sum_of_squares(n_points, residuals_);

        // record current chi^2
        if (chi2_history_size_ < chi2_history_cap_) chi2_history_[chi2_history_size_++] = chi_squared_current;

        // Solve normal equations to get parameter updates
        compute_jacobian(model_func, n_points, x, params, perturbed_params_, n_params, gradients_, jacobian_);
        if (!solve_normal_equations(n_points, n_params, jacobian_, residuals_, delta_params_, damping)) {
            return false; // Failed to solve
        }

        // Propose new parameters
        for (size_t i = 0; i < n_params; ++i) {
            params_updated_[i] = params[i] + delta_params_[i];
        }

        // Compute new chi-squared
        compute_function_values(n_points, x, params_updated_, model_func, function_values_updated_);
        compute_residuals(n_points, y, function_values_updated_, residuals_updated_);
        real chi_squared_new = compute_sum_of_squares(n_points, residuals_updated_);

        // record new chi^2 (proposed)
        if (chi2_history_size_ < chi2_history_cap_) chi2_history_[chi2_history_size_++] = chi_squared_new;

        // Compute norm of parameter changes for convergence check
        real param_change_norm = 0.0;
        for (size_t i = 0; i < n_params; ++i) {
            param_change_norm += delta_params_[i] * delta_params_[i];
        }
        param_change_norm = std::sqrt(param_change_norm);

        // Decision making
        if (chi_squared_new < chi_squared_current) {
            // Accept new parameters
            for (size_t i = 0; i < n_params; ++i) {
                params[i] = params_updated_[i];
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
                params[i] = params_updated_[i];
            }
            // update metrics and store optimized params
            chi_squared_ = chi_squared_new;
            for (size_t i = 0; i < n_params; ++i) fitted_params_[i] = params[i];
            break;
        }

        // increment iteration counter
        ++iterations_;
    }

    // After loop, ensure metrics reflect final params
    // compute final chi-squared and copy params
    compute_function_values(n_points, x, params, model_func, function_values_);
    compute_residuals(n_points, y, function_values_, residuals_);
    chi_squared_ = compute_sum_of_squares(n_points, residuals_);
    for (size_t i = 0; i < n_params; ++i) fitted_params_[i] = params[i];

    return true;
}

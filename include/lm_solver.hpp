#pragma once

#include "types.hpp"
#include "utils.hpp"

// Stateful Levenberg-Marquardt solver object. The class owns reusable
// buffers (function values, Jacobian, temporary vectors) to avoid
// repeated allocations when running fits repeatedly or calling
// intermediate routines.
class LMFit {
public:
	LMFit();
	~LMFit();

	// Ensure internal buffers can hold at least the given sizes. This
	// is cheap if called with the same sizes repeatedly.
	void ensure_capacity(std::size_t n_points, std::size_t n_params);

	// Public API (same logical operations as the previous free functions)
	void compute_function_values(std::size_t n_points, const real* x, const real* params, ModelFuncType model_func, real* output_values);
	void compute_residuals(std::size_t n_points, const real* y, const real* f, real* output_r);
	real compute_sum_of_squares(std::size_t n_points, const real* r);
	void compute_jacobian(ModelFuncType model_func, std::size_t n_points, const real* x, const real* params, real* perturbedParams, std::size_t n_params, real* gradient, real* J);
	bool solve_normal_equations(std::size_t n_points, std::size_t n_params, const real* J, const real* r, real* delta_params, real damping);
	bool levenberg_marquardt_fit(std::size_t n_points, std::size_t n_params, const real* x, const real* y, real* params, ModelFuncType model_func, real tol, std::size_t max_iterations, real damping);

private:
	// reusable buffers (raw pointers to avoid host-side vector allocations)
	std::size_t cap_points_ = 0;
	std::size_t cap_params_ = 0;

	real* function_values_ = nullptr;
	real* function_values_updated_ = nullptr;
	real* gradients_ = nullptr;
	real* residuals_ = nullptr;
	real* residuals_updated_ = nullptr;
	real* perturbed_params_ = nullptr;
	real* jacobian_ = nullptr; // size = cap_points_ * cap_params_
	real* delta_params_ = nullptr;
	real* params_updated_ = nullptr;

	// Temporary matrices/vectors used by solvers
	real* JTJ_ = nullptr; // size = cap_params_ * cap_params_
	real* JTr_ = nullptr; // size = cap_params_
};

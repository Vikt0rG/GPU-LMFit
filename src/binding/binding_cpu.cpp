// Python bindings for CPU-LMFit using pybind11.

#include "../../extern/pybind11/include/pybind11/pybind11.h"
#include "../../extern/pybind11/include/pybind11/numpy.h"
#include "../../extern/pybind11/include/pybind11/stl.h"

#include "../../include/lm_solver.hpp"
#include "../../include/utils.hpp"

namespace py = pybind11;

// Global (process-local) holders for Python callables used as trampolines.
// These are set immediately before a fit and cleared afterwards. This is
// not thread-safe, but is a simple way to bridge ModelDescriptor's raw
// function-pointer API and Python callables.
static py::object g_model_func;
static py::object g_model_deriv;
static std::size_t g_model_n_params = 0;

// Trampoline that matches ModelFuncType: real func(const real* x, const real* params)
static real py_model_func(const real* x, const real* params) {
	py::gil_scoped_acquire gil;
	if (!g_model_func) return (real)0;

	// Call the Python function with a scalar x and a numpy array view of params
	py::array_t<real> py_params(g_model_n_params);
	for (std::size_t i = 0; i < g_model_n_params; ++i) py_params.mutable_data()[i] = params[i];

	py::object res = g_model_func(*x, py_params);
	return res.cast<real>();
}

// Trampoline that matches ModelDerivType: void deriv(const real* x, const real* params, real* out)
static void py_model_deriv(const real* x, const real* params, real* out) {
	py::gil_scoped_acquire gil;
	if (!g_model_deriv) return;

	py::array_t<real> py_params(g_model_n_params);
	for (std::size_t i = 0; i < g_model_n_params; ++i) py_params.mutable_data()[i] = params[i];

	py::object res = g_model_deriv(*x, py_params);
	// Expect an iterable/sequence of length g_model_n_params
	py::array_t<real> arr = py::cast<py::array_t<real>>(res);
	auto r = arr.unchecked<1>();
	for (std::size_t i = 0; i < g_model_n_params; ++i) out[i] = r[i];
}

PYBIND11_MODULE(cpu_lmfit, m) {
	m.doc() = "CPU-LMFit bindings for Levenberg-Marquardt fitting using Python and NumPy";

	py::class_<LMFit>(m, "LMFit")
		.def(py::init<>())
		.def("ensure_capacity", &LMFit::ensure_capacity, "Ensure internal buffer sizes",
			 py::arg("n_points"), py::arg("n_params"))

		// Expose a Python-friendly wrapper for levenberg_marquardt_fit that
		// accepts NumPy arrays and Python callables for the model and
		// optional derivative. The wrapper sets global trampolines used by
		// the C-style ModelDescriptor function pointers.
		.def("levenberg_marquardt_fit",
			[](LMFit &self,
			   py::array_t<real, py::array::c_style | py::array::forcecast> x_arr,
			   py::array_t<real, py::array::c_style | py::array::forcecast> y_arr,
			   py::array_t<real, py::array::c_style | py::array::forcecast> params_arr,
			   py::function model_func,
			   py::object model_deriv, // optional, may be None
			   real tol,
			   std::size_t max_iterations,
			   real damping) {

				// Validate shapes
				if (x_arr.ndim() != 1 || y_arr.ndim() != 1) throw std::runtime_error("x and y must be 1-D arrays");
				if (x_arr.size() != y_arr.size()) throw std::runtime_error("x and y must have the same length");
				if (params_arr.ndim() != 1) throw std::runtime_error("params must be a 1-D array");

				std::size_t n_points = (std::size_t)x_arr.size();
				std::size_t n_params = (std::size_t)params_arr.size();

				// Get raw pointers (forcecast above ensures correct dtype/contiguity)
				const real* x_ptr = x_arr.data();
				const real* y_ptr = y_arr.data();
				real* params_ptr = const_cast<real*>(params_arr.data());

				// Set globals
				g_model_func = model_func;
				if (!model_deriv.is_none()) g_model_deriv = model_deriv.cast<py::function>(); else g_model_deriv = py::object();
				g_model_n_params = n_params;

				// Create ModelDescriptor using trampolines
				ModelDescriptor desc(n_params, &py_model_func, g_model_deriv ? &py_model_deriv : nullptr);

				bool ok = self.levenberg_marquardt_fit(n_points, n_params, x_ptr, y_ptr, params_ptr, desc, tol, max_iterations, damping);

				// Clear globals
				g_model_func = py::object();
				g_model_deriv = py::object();
				g_model_n_params = 0;

				return ok;
			},
			"Run Levenberg-Marquardt fit using Python callables and NumPy arrays",
			py::arg("x"), py::arg("y"), py::arg("params"), py::arg("model_func"), py::arg("model_deriv") = py::none(),
			py::arg("tol") = (real)1e-6, py::arg("max_iterations") = (std::size_t)100, py::arg("damping") = (real)1e-3)

		.def("get_chi_squared", &LMFit::get_chi_squared)
		.def("get_iterations", &LMFit::get_iterations)
		.def("print_fit_metrics", &LMFit::print_fit_metrics)

		// Return optimized params as a new NumPy array of length n_params
		.def("get_optimized_params", [](LMFit &self, std::size_t n_params) {
			py::array_t<real> out(n_params);
			real* out_ptr = out.mutable_data();
			self.copy_optimized_params(out_ptr, n_params);
			return out;
		}, py::arg("n_params"))

		// Return chi2 history as a NumPy array
		.def("get_chi2_history", [](LMFit &self) {
			std::size_t sz = self.get_chi2_history_size();
			py::array_t<real> out(sz);
			if (sz == 0) return out;
			const real* src = self.get_chi2_history_ptr();
			real* dst = out.mutable_data();
			for (std::size_t i = 0; i < sz; ++i) dst[i] = src[i];
			return out;
		})
		;
}

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <type_traits>
#include "../../include/utils.hpp"

// Device-only Levenberg-Marquardt implementation.

// Helper to check cuBLAS calls
static void cublas_check(cublasStatus_t stat, const char* where) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error at ") + where + ": " + std::to_string((int)stat));
    }
}

// cuSOLVER check helper
static void cusolver_check(cusolverStatus_t stat, const char* where) {
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSOLVER error at ") + where + ": " + std::to_string((int)stat));
    }
}

#define MAX_PARAMS 64

// Evaluate model at each x (device)
template <typename Model>
__global__ void eval_model_kernel(Model model, const real* x, const real* params, real* out_f, size_t n_points) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    out_f[idx] = model.model(&x[idx], params);
}

// Jacobian kernel when analytic derivative is available
template <typename Model>
__global__ void jacobian_analytic_kernel(Model model, const real* x, const real* params, real* J, size_t n_points, size_t n_params) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    real grad[MAX_PARAMS];
    model.derivative(&x[i], params, grad);
    for (size_t j = 0; j < n_params; ++j) {
        J[i * n_params + j] = grad[j];
    }
}

// Numeric forward difference Jacobian on device (per (i,j) thread pattern)
template <typename Model>
__global__ void jacobian_numeric_kernel(Model model, const real* x, const real* params, real* J, size_t n_points, size_t n_params) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // point index
    size_t j = blockIdx.y * blockDim.y + threadIdx.y; // param index
    if (i >= n_points || j >= n_params) return;
    if (n_params > MAX_PARAMS) return; // guard

    real local_p[MAX_PARAMS];
    for (size_t k = 0; k < n_params; ++k) local_p[k] = params[k];

    real f0 = model.model(&x[i], params);
    real h = (real)(1e-6) * (real)(fabsf(local_p[j]) + 1.0f);
    local_p[j] += h;
    real fp = model.model(&x[i], local_p);
    J[i * n_params + j] = (fp - f0) / h;
}

// Device utility kernels (moved here so this file is self-contained)
// calculate_residuals: r = y - f
__global__ void calculate_residuals(const real* y, const real* f, real* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    r[idx] = y[idx] - f[idx];
}

// accumulate_JTJ_JTr: compute JTJ (n_params x n_params) and JTr (n_params)
// J is row-major: J[i * n_params + p] == derivative at point i for param p
__global__ void accumulate_JTJ_JTr(const real* J, const real* r, real* JTJ, real* JTr, size_t n_points, size_t n_params) {
    size_t p = blockIdx.x * blockDim.x + threadIdx.x;
    size_t q = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= n_params || q >= n_params) return;

    real sum = 0;
    for (size_t i = 0; i < n_points; ++i) {
        real Jip = J[i * n_params + p];
        real Jiq = J[i * n_params + q];
        sum += Jip * Jiq;
    }
    JTJ[p * n_params + q] = sum;

    // let threads with q == 0 also compute JTr for their p index to avoid a second kernel
    if (q == 0) {
        real sum2 = 0;
        for (size_t i = 0; i < n_points; ++i) {
            sum2 += J[i * n_params + p] * r[i];
        }
        JTr[p] = sum2;
    }
}

// add_damping: add damping to the diagonal entries of JTJ
__global__ void add_damping(real* JTJ, real damping, size_t n_params) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_params) return;
    JTJ[idx * n_params + idx] += damping;
}

// Templated device LM driver
template <typename Model>
bool gpu_levenberg_marquardt_fit_device(
    size_t n_points,
    size_t n_params,
    const real* x_host,
    const real* y_host,
    real* params_host,
    const Model& model,
    real tol,
    size_t max_iterations,
    real damping
) {
    if (n_params > MAX_PARAMS) return false;

    // cuBLAS / cuSOLVER handles
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cublas_check(cublasCreate(&cublas), "cublasCreate device LM");
    if (cusolverDnCreate(&cusolver) != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(cublas);
        return false;
    }

    // allocate device buffers
    real *d_x = nullptr, *d_y = nullptr, *d_params = nullptr, *d_f = nullptr, *d_r = nullptr, *d_J = nullptr;
    size_t x_bytes = n_points * sizeof(real);
    size_t y_bytes = n_points * sizeof(real);
    size_t params_bytes = n_params * sizeof(real);
    size_t J_bytes = n_points * n_params * sizeof(real);

    cudaError_t cerr = cudaMalloc((void**)&d_x, x_bytes);
    if (cerr != cudaSuccess) return false;
    cerr = cudaMalloc((void**)&d_y, y_bytes); if (cerr != cudaSuccess) goto fail_cleanup;
    cerr = cudaMalloc((void**)&d_params, params_bytes); if (cerr != cudaSuccess) goto fail_cleanup;
    cerr = cudaMalloc((void**)&d_f, y_bytes); if (cerr != cudaSuccess) goto fail_cleanup;
    cerr = cudaMalloc((void**)&d_r, y_bytes); if (cerr != cudaSuccess) goto fail_cleanup;
    cerr = cudaMalloc((void**)&d_J, J_bytes); if (cerr != cudaSuccess) goto fail_cleanup;

    if (cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail_cleanup;
    if (cudaMemcpy(d_y, y_host, y_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail_cleanup;
    if (cudaMemcpy(d_params, params_host, params_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail_cleanup;

    // device outputs for JTJ and JTr
    real* d_JTJ = nullptr;
    real* d_JTr = nullptr;
    size_t JTJ_bytes = n_params * n_params * sizeof(real);
    size_t JTr_bytes = n_params * sizeof(real);
    if (cudaMalloc((void**)&d_JTJ, JTJ_bytes) != cudaSuccess) goto fail_cleanup;
    if (cudaMalloc((void**)&d_JTr, JTr_bytes) != cudaSuccess) goto fail_cleanup;

    int* d_devInfo = nullptr;
    if (cudaMalloc((void**)&d_devInfo, sizeof(int)) != cudaSuccess) goto fail_cleanup;

    using RealT = typename std::conditional<std::is_same<real, float>::value, float, double>::type;

    // main LM loop
    size_t iter = 0;
    while (iter++ < max_iterations) {
        int block = 256;
        int grid = (int)((n_points + block - 1) / block);
        eval_model_kernel<Model><<<grid, block>>>(model, d_x, d_params, d_f, n_points);
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        int blk = 256;
        int grd = (int)((n_points + blk - 1) / blk);
        calculate_residuals<<<grd, blk>>>(d_y, d_f, d_r, n_points);
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        // jacobian
        struct has_derivative_impl {
            template <typename T>
            static auto test(int) -> decltype(std::declval<T>().derivative(std::declval<const real*>(), std::declval<const real*>(), std::declval<real*>()), std::true_type());
            template <typename> static std::false_type test(...);
        };
        constexpr bool has_derivative = decltype(has_derivative_impl::test<Model>(0))::value;
        if (has_derivative) {
            int jb = (int)((n_points + 255) / 256);
            jacobian_analytic_kernel<Model><<<jb, 256>>>(model, d_x, d_params, d_J, n_points, n_params);
        } else {
            dim3 block2(16, 16);
            dim3 grid2((n_points + block2.x - 1) / block2.x, (n_params + block2.y - 1) / block2.y);
            jacobian_numeric_kernel<Model><<<grid2, block2>>>(model, d_x, d_params, d_J, n_points, n_params);
        }
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        dim3 blockA(16,16);
        dim3 gridA((n_params + blockA.x - 1) / blockA.x, (n_params + blockA.y - 1) / blockA.y);
        accumulate_JTJ_JTr<<<gridA, blockA>>>(d_J, d_r, d_JTJ, d_JTr, n_points, n_params);
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        int db = (int)((n_params + 255) / 256);
        add_damping<<<db, 256>>>(d_JTJ, damping, n_params);
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        // Solve JTJ * delta = JTr on device using cuSOLVER (Cholesky)
        int n = (int)n_params;
        int lwork = 0;
        RealT* d_work = nullptr;

        // determine workspace size and allocate
        CusolverFuncs<RealT>::potrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, n, (RealT*)d_JTJ, n, &lwork);
        if (lwork > 0) cudaMalloc((void**)&d_work, lwork * sizeof(RealT));

        cusolver_check(CusolverFuncs<RealT>::potrf(cusolver, CUBLAS_FILL_MODE_LOWER, n, (RealT*)d_JTJ, n, d_work, lwork, d_devInfo), "potrf");
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        int devInfo_h = 0;
        if (cudaMemcpy(&devInfo_h, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail_cleanup;
        if (devInfo_h != 0) goto fail_cleanup; // factorization failed

        cusolver_check(CusolverFuncs<RealT>::potrs(cusolver, CUBLAS_FILL_MODE_LOWER, n, 1, (RealT*)d_JTJ, n, (RealT*)d_JTr, n, d_devInfo), "potrs");
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        // update params: params += delta (JTr contains delta)
        RealT alpha = (RealT)1.0;
        cublasStatus_t axpy_stat = cublasAxpy(cublas, n, &alpha, (RealT*)d_JTr, 1, (RealT*)d_params, 1);
        cublas_check(axpy_stat, "cublasAxpy update params");

        // evaluate updated model and residuals to compute chi2 and convergence
        eval_model_kernel<Model><<<grid, block>>>(model, d_x, d_params, d_f, n_points);
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        calculate_residuals<<<grd, blk>>>(d_y, d_f, d_r, n_points);
        if (cudaGetLastError() != cudaSuccess) goto fail_cleanup;
        if (cudaDeviceSynchronize() != cudaSuccess) goto fail_cleanup;

        // copy residuals and delta back to host to check convergence
        std::vector<real> residuals_host(n_points);
        if (cudaMemcpy(residuals_host.data(), d_r, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail_cleanup;
        real chi2_new = 0.0f;
        for (size_t i = 0; i < n_points; ++i) chi2_new += residuals_host[i] * residuals_host[i];

        static real chi2_prev = std::numeric_limits<real>::infinity();
        if (chi2_new < chi2_prev) damping *= (real)0.1; else damping *= (real)10.0;
        chi2_prev = chi2_new;

        std::vector<real> delta_host(n_params);
        if (cudaMemcpy(delta_host.data(), d_JTr, JTr_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail_cleanup;
        real param_change_norm = 0.0f;
        for (size_t i = 0; i < n_params; ++i) param_change_norm += delta_host[i] * delta_host[i];
        param_change_norm = std::sqrt(param_change_norm);

        if (std::abs(chi2_prev - chi2_new) < tol || param_change_norm < tol) break;
    }

    // copy params back
    if (cudaMemcpy(params_host, d_params, params_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail_cleanup;

    // cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_params); cudaFree(d_f); cudaFree(d_r); cudaFree(d_J); cudaFree(d_JTJ); cudaFree(d_JTr); cudaFree(d_devInfo);
    if (cusolver) cusolverDnDestroy(cusolver);
    if (cublas) cublasDestroy(cublas);
    return true;

fail_cleanup:
    // best-effort cleanup
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_params) cudaFree(d_params);
    if (d_f) cudaFree(d_f);
    if (d_r) cudaFree(d_r);
    if (d_J) cudaFree(d_J);
    if (d_JTJ) cudaFree(d_JTJ);
    if (d_JTr) cudaFree(d_JTr);
    if (d_devInfo) cudaFree(d_devInfo);
    if (cusolver) cusolverDnDestroy(cusolver);
    if (cublas) cublasDestroy(cublas);
    return false;
}

// Thin host-visible wrapper that deduces model type and calls device driver.
// This function keeps the same name as before but is a template; callers should
// instantiate it with the proper Model functor.
template <typename Model>
bool gpu_levenberg_marquardt_fit(
    size_t n_points,
    size_t n_params,
    const real* x,
    const real* y,
    real* params,
    const Model& model,
    real tol,
    size_t max_iterations,
    real damping
) {
    return gpu_levenberg_marquardt_fit_device<Model>(n_points, n_params, x, y, params, model, tol, max_iterations, damping);
}
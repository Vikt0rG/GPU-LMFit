#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                    \
    do {                                                                                                    \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess) {                                                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)

__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    printf("CUDA smoke test start\n");

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global mem: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }

    // Use device 0 for the functional test
    CUDA_CHECK(cudaSetDevice(0));

    // Simple vector add test
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    float *h_a = nullptr, *h_b = nullptr, *h_c = nullptr;
    // Host pinned allocation to ensure proper pageable/pinned paths are exercised
    CUDA_CHECK(cudaMallocHost((void**)&h_a, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_b, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_c, bytes));

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
        h_c[i] = 0.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    addKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Validate results (sample a few values and check a checksum)
    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += h_c[i];
    // Expected sum: sum_i (i + 2i) = sum_i (3i) = 3 * N*(N-1)/2
    double expected = 3.0 * (double)N * (double)(N - 1) / 2.0;

    const double rel_err = fabs((checksum - expected) / expected);
    printf("Checksum: %.0f, Expected: %.0f, Relative error: %.3e\n", checksum, expected, rel_err);

    bool ok = rel_err < 1e-6;
    if (ok) {
        printf("Result verification: PASSED\n");
    } else {
        printf("Result verification: FAILED\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("CUDA smoke test end\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

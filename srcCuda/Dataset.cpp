#include "../include/Dataset.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#define CHECK_CUDA_ERROR(call)                               \
    {                                                        \
        cudaError_t err = call;                              \
        if (err != cudaSuccess)                              \
        {                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    }

namespace Neural {

// Kernel para encontrar o máximo e o mínimo de uma coluna
__global__ void findMinMaxKernel(double *data, double *min, double *max, int rows, int col) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        atomicMin(min, data[idx * col]);
        atomicMax(max, data[idx * col]);
    }
}

// Kernel para normalizar os valores
__global__ void normalizeKernel(double *data, double min, double max, int rows, int col) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        data[idx * col] = (data[idx * col] - min) / (max - min);
    }
}

void Dataset::normalize(std::vector<std::vector<double>> &v) {
    int rows = v.size();
    int cols = v[0].size();

    // Alocar memória na GPU
    double *d_data, *d_min, *d_max;
    double min = v[0][0], max = v[0][0];

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_min, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_max, sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, &v[0][0], rows * cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_min, &min, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_max, &max, sizeof(double), cudaMemcpyHostToDevice));

    // Lançar kernel para encontrar min e max
    int threads_per_block = 256;
    int blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;
    findMinMaxKernel<<<blocks_per_grid, threads_per_block>>>(d_data, d_min, d_max, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copiar min e max para CPU
    CHECK_CUDA_ERROR(cudaMemcpy(&min, d_min, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost));

    // Lançar kernel para normalizar
    normalizeKernel<<<blocks_per_grid, threads_per_block>>>(d_data, min, max, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copiar resultados normalizados de volta para a CPU
    CHECK_CUDA_ERROR(cudaMemcpy(&v[0][0], d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));

    // Liberar memória na GPU
    cudaFree(d_data);
    cudaFree(d_min);
    cudaFree(d_max);
}

}

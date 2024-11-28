#include "../include/Network.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

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

// CUDA Kernel para calcular a soma ponderada na camada oculta
__global__ void forwardHiddenKernel(double *input, double *weight_input, double *sum_input_weight, int input_size, int hidden_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hidden_size) {
        double sum = 0.0;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weight_input[j * hidden_size + idx];
        }
        sum_input_weight[idx] = sum;
    }
}

// CUDA Kernel para aplicar a função de ativação (sigmoid)
__global__ void applySigmoidKernel(double *values, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        values[idx] = 1.0 / (1.0 + exp(-values[idx]));
    }
}

Network::Network() {}

Network::Network(std::vector<std::vector<double>> user_input, std::vector<std::vector<double>> user_output) {
    setInput(user_input);
    setOutput(user_output);
    output_layer_size = 3;
}

void Network::run() {
    // Preparação para paralelizar o cálculo na GPU
    for (unsigned int data_row = 0; data_row < input.size(); data_row++) {
        ForwardPropagation forward = forwardPropagation(input[data_row]);
        hitRateCount(forward.output, data_row);
    }
    hitRateCalculate();
}

Network::ForwardPropagation Network::forwardPropagation(std::vector<double> input_line) {
    input_line.push_back(1); // Adicionar bias

    // Alocação de memória na GPU
    double *d_input, *d_weight_input, *d_sum_input_weight;
    int input_size = input_layer_size;
    int hidden_size = hidden_layer_size;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight_input, input_size * hidden_size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum_input_weight, hidden_size * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input_line.data(), input_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight_input, weight_input[0].data(), input_size * hidden_size * sizeof(double), cudaMemcpyHostToDevice));

    // Lançar kernel para forwardPropagation
    int threads_per_block = 256;
    int blocks_per_grid = (hidden_size + threads_per_block - 1) / threads_per_block;
    forwardHiddenKernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_weight_input, d_sum_input_weight, input_size, hidden_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Aplicar a função sigmoid
    applySigmoidKernel<<<blocks_per_grid, threads_per_block>>>(d_sum_input_weight, hidden_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copiar resultados de volta para a CPU
    std::vector<double> sum_input_weight(hidden_size);
    CHECK_CUDA_ERROR(cudaMemcpy(sum_input_weight.data(), d_sum_input_weight, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Liberar memória na GPU
    cudaFree(d_input);
    cudaFree(d_weight_input);
    cudaFree(d_sum_input_weight);

    // Montar e retornar os resultados
    ForwardPropagation forward(hidden_size, output_layer_size);
    forward.sum_input_weight = sum_input_weight;

    return forward;
}

// Outros métodos seguem a mesma ideia de paralelização, adaptando o cálculo para CUDA
}

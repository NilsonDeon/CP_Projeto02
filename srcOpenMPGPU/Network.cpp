#include "../include/Network.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>

namespace Neural {

Network::ForwardPropagation Network::forwardPropagation(std::vector<double> input_line) {
    input_line.push_back(1); // Adicionar bias

    std::vector<double> sum_input_weight(hidden_layer_size, 0.0);

    // Forward Propagation (na GPU)
    #pragma omp target teams distribute parallel for map(to: input_line[0:input_line.size()], weight_input[0:input_layer_size][0:hidden_layer_size]) map(from: sum_input_weight[0:hidden_layer_size])
    for (int i = 0; i < hidden_layer_size; i++) {
        for (int j = 0; j < input_layer_size; j++) {
            sum_input_weight[i] += input_line[j] * weight_input[j][i];
        }
    }

    // Aplicar a função sigmoid (na GPU)
    #pragma omp target teams distribute parallel for map(tofrom: sum_input_weight[0:hidden_layer_size])
    for (int i = 0; i < hidden_layer_size; i++) {
        sum_input_weight[i] = 1.0 / (1.0 + exp(-sum_input_weight[i]));
    }

    // Montar e retornar os resultados
    ForwardPropagation forward(hidden_layer_size, output_layer_size);
    forward.sum_input_weight = sum_input_weight;

    return forward;
}

}

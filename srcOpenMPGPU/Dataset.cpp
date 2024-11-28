#include "../include/Dataset.hpp"
#include <iostream>
#include <algorithm>
#include <omp.h>

namespace Neural {

void Dataset::normalize(std::vector<std::vector<double>> &v) {
    int rows = v.size();
    int cols = v[0].size();

    double min = v[0][0], max = v[0][0];

    // Encontrar o mínimo e o máximo (na GPU)
    #pragma omp target teams distribute parallel for map(to: v[0:rows][0:cols]) map(tofrom: min, max)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            #pragma omp critical
            {
                if (v[i][j] < min) min = v[i][j];
                if (v[i][j] > max) max = v[i][j];
            }
        }
    }

    // Normalizar os valores (na GPU)
    #pragma omp target teams distribute parallel for map(to: v[0:rows][0:cols], min, max)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            v[i][j] = (v[i][j] - min) / (max - min);
        }
    }
}

}

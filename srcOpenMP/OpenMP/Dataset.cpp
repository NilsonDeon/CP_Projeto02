#include "../../include/Dataset.hpp"
#include <iomanip>
#include <omp.h>

namespace Neural {

Dataset::Dataset() {
}

void Dataset::saveOutputLog() {
}

void Dataset::printMatrix(vector<vector<double>> v) {
#pragma omp parallel for num_threads(4)
    for (unsigned int i = 0; i < v.size(); i++) {
        for (unsigned int j = 0; j < v[i].size(); j++) {
            cout << fixed << setprecision(2) << v[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

void Dataset::printVector(vector<double> v) {
#pragma omp parallel for num_threads(4)
    for (unsigned int i = 0; i < v.size(); i++) {
        cout << fixed << setprecision(2) << v[i] << "\t";
    }
    cout << endl;
}

/*
 * Load input and output in the same file
 */
void Dataset::loadInputOutputData(int n_input, int n_output, string file) {
    n_inputs = n_input;
    n_outputs = n_output;

    ifstream input;
    input.open(file);
    double data;

    if (!input.is_open()) {
        cout << "FALHA ARQUIVO" << endl;
    } else {
        n_rows = int(count(istreambuf_iterator<char>(input), istreambuf_iterator<char>(), '\n')) + 1;

        input.clear();
        input.seekg(0, input.beg);

        input_data.resize(n_rows);
        output_data.resize(n_rows);

#pragma omp parallel for num_threads(4)
        for (int i = 0; i < n_rows; i++) {
            vector<double> temp_input, temp_output;
            for (int j = 0; j < n_inputs; j++) {
                input >> data;
                temp_input.push_back(data);
            }
            for (int j = 0; j < n_outputs; j++) {
                input >> data;
                temp_output.push_back(data);
            }
#pragma omp critical
            {
                input_data[i] = temp_input;
                output_data[i] = temp_output;
            }
        }

        normalize(input_data);
    }

    input.close();
}

void Dataset::normalize(vector<vector<double>> v) {
#pragma omp parallel for num_threads(4)
    for (unsigned int i = 0; i < v[0].size(); i++) {
        double max = v[0][i];
        double min = v[0][i];

#pragma omp parallel for reduction(max : max) reduction(min : min) num_threads(4)
        for (unsigned int j = 0; j < v.size(); j++) {
            if (max < v[j][i]) max = v[j][i];
            if (min > v[j][i]) min = v[j][i];
        }

#pragma omp parallel for num_threads(4)
        for (unsigned int j = 0; j < v.size(); j++) {
            input_data[j][i] = (v[j][i] - min) / (max - min);
        }
    }
}

vector<vector<double>> Dataset::getInput() {
    return input_data;
}

vector<vector<double>> Dataset::getOutput() {
    return output_data;
}

}

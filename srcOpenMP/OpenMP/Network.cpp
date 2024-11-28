#include "../../include/Network.hpp"

#define NUM_THREADS 4

namespace Neural
{

    Network::Network()
    {
    }

    Network::Network(vector<vector<double>> user_input, vector<vector<double>> user_output)
    {
        setInput(user_input);
        setOutput(user_output);
        output_layer_size = 3;
    }

    void Network::setParameter(int user_max_epoch, int user_desired_percent, double user_error_tolerance, double user_learning_rate, int user_hidden_layer_size)
    {
        setMaxEpoch(user_max_epoch);
        setLearningRate(user_learning_rate);
        setErrorTolerance(user_error_tolerance);
        setDesiredPercent(user_desired_percent);
        setHiddenLayerSize(user_hidden_layer_size);
        best_network.epoch = max_epoch;

        initializeWeight();
    }

    void Network::run()
    {
#pragma omp parallel for num_threads(NUM_THREADS)
        for (unsigned int data_row = 0; data_row < input.size(); data_row++)
        {
            ForwardPropagation forward = forwardPropagation(input[data_row]);
            int local_correct = 0;
            for (int i = 0; i < output_layer_size; i++)
            {
                if (abs(forward.output[i] - output[data_row][i]) < error_tolerance)
                    local_correct++;
            }
#pragma omp atomic
            correct_output += local_correct;
        }
        hitRateCalculate();
    }

    void Network::trainingClassification()
    {
        for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++)
        {
#pragma omp parallel for num_threads(NUM_THREADS)
            for (unsigned int data_row = 0; data_row < input.size(); data_row++)
            {
                ForwardPropagation forward = forwardPropagation(input[data_row]);
                backPropagation(forward, input[data_row], output[data_row]);
            }
            run();
        }

        cout << "Hidden Layer Size: " << hidden_layer_size
             << "\tLearning Rate: " << learning_rate
             << "\tHit Percent: " << hit_percent << "%"
             << "\tEpoch: " << epoch << endl;
    }

    void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase)
    {
        for (hidden_layer_size = 3; hidden_layer_size <= hidden_layer_limit; hidden_layer_size++)
        {
            for (learning_rate = learning_rate_increase; learning_rate <= 1; learning_rate += learning_rate_increase)
            {
                initializeWeight();
                trainingClassification();
                if (epoch < best_network.epoch)
                {
                    best_network.epoch = epoch;
                    best_network.learning_rate = learning_rate;
                    best_network.hidden_layer = hidden_layer_size;
                    best_network.weight_input = weight_input;
                    best_network.weight_output = weight_output;
                }
            }
        }

        cout << "Best Network --> Hidden Layer Size: " << best_network.hidden_layer
             << "\tLearning Rate: " << best_network.learning_rate
             << "\tEpoch: " << best_network.epoch << endl;

        epoch = best_network.epoch;
        learning_rate = best_network.learning_rate;
        hidden_layer_size = best_network.hidden_layer;
        weight_input = best_network.weight_input;
        weight_output = best_network.weight_output;
    }

    Network::ForwardPropagation Network::forwardPropagation(vector<double> input_line)
    {
        input_line.push_back(1); // bias

        ForwardPropagation forward(hidden_layer_size, output_layer_size);

#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
        for (int i = 0; i < hidden_layer_size; i++)
        {
            for (int j = 0; j < input_layer_size; j++)
            {
                forward.sum_input_weight[i] += input_line[j] * weight_input[j][i];
            }
        }

        for (int i = 0; i < hidden_layer_size; i++)
        {
            forward.sum_input_weight_ativation.push_back(sigmoid(forward.sum_input_weight[i]));
        }

#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
        for (int i = 0; i < output_layer_size; i++)
        {
            for (int j = 0; j < hidden_layer_size; j++)
            {
                forward.sum_output_weigth[i] += forward.sum_input_weight_ativation[j] * weight_output[j][i];
            }
        }

        for (int i = 0; i < output_layer_size; i++)
        {
            forward.output.push_back(sigmoid(forward.sum_output_weigth[i]));
        }

        return forward;
    }

    void Network::backPropagation(ForwardPropagation forward, vector<double> input_line, vector<double> output_line)
    {
        input_line.push_back(1); // bias

        BackPropagation back(hidden_layer_size);

        for (int i = 0; i < output_layer_size; i++)
        {
            back.delta_output_sum.push_back((output_line[i] - forward.output[i]) * sigmoidPrime(forward.sum_output_weigth[i]));
        }

        for (int i = 0; i < hidden_layer_size; i++)
        {
            for (int j = 0; j < output_layer_size; j++)
            {
                back.delta_input_sum[i] += back.delta_output_sum[j] * weight_output[i][j];
            }
            back.delta_input_sum[i] *= sigmoidPrime(forward.sum_input_weight[i]);
        }

#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
        for (unsigned int i = 0; i < weight_output.size(); i++)
        {
            for (unsigned int j = 0; j < weight_output[i].size(); j++)
            {
                weight_output[i][j] += back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;
            }
        }

#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
        for (unsigned int i = 0; i < weight_input.size(); i++)
        {
            for (unsigned int j = 0; j < weight_input[i].size(); j++)
            {
                weight_input[i][j] += back.delta_input_sum[j] * input_line[i] * learning_rate;
            }
        }
    }

    void Network::hitRateCalculate()
    {
        hit_percent = (correct_output * 100) / (output.size() * output_layer_size);
        correct_output = 0;
    }

    void Network::initializeWeight()
    {
        weight_input.assign(input_layer_size, vector<double>(hidden_layer_size));
        weight_output.assign(hidden_layer_size, vector<double>(output_layer_size));

        srand((unsigned int)time(0));

#pragma omp parallel for num_threads(NUM_THREADS)
        for (unsigned int i = 0; i < weight_input.size(); i++)
        {
            for (int j = 0; j < hidden_layer_size; j++)
            {
                weight_input[i][j] = ((double)rand() / (RAND_MAX));
            }
        }

#pragma omp parallel for num_threads(NUM_THREADS)
        for (unsigned int i = 0; i < weight_output.size(); i++)
        {
            for (int j = 0; j < output_layer_size; j++)
            {
                weight_output[i][j] = ((double)rand() / (RAND_MAX));
            }
        }

        hit_percent = 0;
        correct_output = 0;
    }

    double Network::sigmoid(double z)
    {
        return 1 / (1 + exp(-z));
    }

    double Network::sigmoidPrime(double z)
    {
        return exp(-z) / (pow(1 + exp(-z), 2));
    }

    void Network::setMaxEpoch(int m)
    {
        max_epoch = m;
    }

    void Network::setDesiredPercent(int d)
    {
        desired_percent = d;
    }

    void Network::setHiddenLayerSize(int h)
    {
        hidden_layer_size = h;
    }

    void Network::setLearningRate(double l)
    {
        learning_rate = l;
    }

    void Network::setErrorTolerance(double e)
    {
        error_tolerance = e;
    }

    void Network::setInput(vector<vector<double>> i)
    {
        input = i;
        input_layer_size = i[0].size() + 1; // +1 bias
    }

    void Network::setOutput(vector<vector<double>> o)
    {
        output = o;
        output_layer_size = o[0].size();
    }

}

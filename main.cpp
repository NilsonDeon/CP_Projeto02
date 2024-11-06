#include "include/Network.hpp"
#include "include/Dataset.hpp"

int main(){

	Neural::Dataset data_learning;
		data_learning.loadInputOutputData(4, 3, "database/iris.txt");

	vector<vector<double>> input = data_learning.getInput();
	vector<vector<double>> output = data_learning.getOutput();

	int max_epochs = 1000;                // numero maximo de epocas que serao testadas
	int desired_hit_percent = 95;         // numero minimo de porcentagem de acertos aceitados
	double error_tolerance = 0.05;        // tolerancia de erro. se for maior que ele, e considerado como previsao errada
	int hidden_layer_limit = 15;          // numero maximo de camadas escondidas
	double learning_rate_increase = 0.25; // aumento da taxa de aprendizado (quanto em quanto, de 0.0 a 1.0)

	Neural::Network neural_network(input, output);
		neural_network.setParameter(max_epochs, desired_hit_percent, error_tolerance);
		neural_network.autoTraining(hidden_layer_limit, learning_rate_increase);
		neural_network.run();

	return 0;
}
#include "include/Network.hpp"
#include "include/Dataset.hpp"

int main(){

	Neural::Dataset data_learning;
		data_learning.loadInputOutputData(4, 3, "database/iris.txt");

	vector<vector<double>> input = data_learning.getInput();
	vector<vector<double>> output = data_learning.getOutput();

	Neural::Network neural_network(input, output);
		neural_network.setParameter(1000, 90, 0.05);
		neural_network.autoTraining(20, 0.2);
		neural_network.run();
}
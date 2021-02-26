#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define crelu(x)(x>0.0?x:x*0.1)
#define creluPrime(x)(x>0.0?1.0:0.1)

const int networkLayout[] = {784, 28, 28, 10};
const int networkSize = 4;
const double learningRate = 0.05;
const double expectedOutputs[10][10] = {
	{1,0,0,0,0,0,0,0,0,0},
	{0,1,0,0,0,0,0,0,0,0},
	{0,0,1,0,0,0,0,0,0,0},
	{0,0,0,1,0,0,0,0,0,0},
	{0,0,0,0,1,0,0,0,0,0},
	{0,0,0,0,0,1,0,0,0,0},
	{0,0,0,0,0,0,1,0,0,0},
	{0,0,0,0,0,0,0,1,0,0},
	{0,0,0,0,0,0,0,0,1,0},
	{0,0,0,0,0,0,0,0,0,1}
};

const int trainingDatasetSize = 60000;
const int testingDatasetSize = 10000;

/**
 * @brief This function loads in adn normilises the training data and its labels
 * @param data - the destination for the training data
 * @param lables - the destination for the training data labels
 * @param filePath - the file path that the data is stored at
 * @return nothing
 */
void loadData(double** data, char* labels, const char* filePath) {
	FILE* f;
	char tempStr[4096];
	const char delim[2] = ",";
	char* token;
	
	f = fopen(filePath, "r");
	fscanf(f, "%*[^\n]");		//reads the first line without storing it anywhere (jumps to line 2)
	
	printf("Loading input data...\n");

	int i = 0;
	while (fscanf(f, "%s", tempStr) != EOF) {
		token = strtok(tempStr, delim);
		labels[i] = token[0];
		data[i] = calloc(784, sizeof(double));

		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++) {
				token = strtok(NULL, delim);
				int temp;
				sscanf(token, "%d", &temp);
				data[i][28*j+k] = temp/255.0;
			}
		}
		i++;
	}

	fclose(f);
	printf("Input data loaded!\n");
}

/**
 * @brief This function creates a new network "brain" with random weights and biases
 * 	  between -2 and 2
 * @param weights - destination for the new weights.
 * @param biases - destiantion for the new biases
 * @return nothing
 */
void initialiseNetworkBrain(double*** weights, double** biases) {
	srand(time(NULL));
	for (int i = 0; i < networkSize - 1; i++) {
		weights[i] = calloc(networkLayout[i], sizeof(double*));
		biases[i] = calloc(networkLayout[i+1], sizeof(double));

		//randomise the weights
		for (int j = 0; j < networkLayout[i]; j++) {
			weights[i][j] = calloc(networkLayout[i+1], sizeof(double));
			for (int k = 0; k < networkLayout[i+1]; k++) 
				weights[i][j][k] = (((double)rand() / RAND_MAX) * 4) - 2;
		}

		//randomise the biases
		for (int j = 0; j < networkLayout[i+1]; j++) 
			biases[i][j] = (((double)rand() / RAND_MAX) * 4) - 2;
	}
}

/**
 * @brief This function writes the network to a file for later use
 * @param weights - the weights for each nodal connection in the network
 * @param biases -  the bias for each node in the network
 * @param filePath - the file that will be written to
 * @return nothing
 */
void saveBrain(double*** weights, double** biases, const char* filePath) {
	FILE* f = fopen(filePath, "w");

	//writes the network structure to a file
	for (int i = 0; i < networkSize - 1; i++) 
		fprintf(f, "%d,", networkLayout[i]);
	fprintf(f, "%d\n", networkLayout[networkSize-1]);

	//writes the weight values to the file
	for (int i = 0; i < networkSize - 1; i++) 
		for (int j = 0; j < networkLayout[i]; j++) 
			for (int k = 0; k < networkLayout[i+1]; k++) 
				fprintf(f, "%.100lf\n", weights[i][j][k]);
	//writes the biases to the file
	for (int i = 0; i < networkSize - 1; i++) 
		for (int j = 0; j < networkLayout[i+1]; j++) 
			fprintf(f, "%.100lf\n", biases[i][j]);
	fclose(f);
}

/**
 * @brief This function reads the network from a file
 * @param weights - stores the weights for each nodal connection in the network
 * @param biases -  stores the bias for each node in the network
 * @param filePath - the file that will be written to
 * @return nothing
 */
void loadBrain(double*** weights, double** biases, const char* filePath) {
	FILE* f = fopen(filePath, "r");

	char tempStr[4096];
	const char delim[2] = ",";
	char* token;

	fscanf(f, "%s", tempStr);

	int i = 0;
	for (token = strtok(tempStr, delim); token && *token; token = strtok(NULL, delim)) {
		if (atoi(token) != networkLayout[i]) {
			printf("Error loading network, dimensions do not match\n");
			exit(1);
		}
		i++;
	}


	char* e;
	//writes the weight values to the file
	for (int i = 0; i < networkSize - 1; i++) {
		weights[i] = calloc(networkLayout[i], sizeof(double*));
		for (int j = 0; j < networkLayout[i]; j++) {
			weights[i][j] = calloc(networkLayout[i+1], sizeof(double));
			for (int k = 0; k < networkLayout[i+1]; k++) {
				fscanf(f, "%s", tempStr);
				weights[i][j][k] = strtod(tempStr, &e);
			}
		}
	}

	//writes the biases to the file
	for (int i = 0; i < networkSize - 1; i++) {
		biases[i] = calloc(networkLayout[i+1], sizeof(double));
		for (int j = 0; j < networkLayout[i+1]; j++) {
			fscanf(f, "%s", tempStr);
			biases[i][j] = strtod(tempStr, &e);
		}
	}
	fclose(f);
}

/**
 * @brief This function frees the memory of the weights and biases
 * @param weights - the networks weights
 * @param biases - the node biases
 * @return nothing
 */
void destroyBrainData(double*** weights, double** biases) {
	
	for (int i = 0; i < networkSize - 1; i++) {
		for (int j = 0; j < networkLayout[i]; j++) 
			free(weights[i][j]);
		free(weights[i]);
		free(biases[i]);
	}
}

/**
 * @brief The function allocated the memory for the change in weights and biases
 * @param deltaWeights - stores the change in weights for each of the images in the batch
 * @param deltaBiases - stores the change in biases for each of the imges in the batch
 * @param batchSize - stores the size of the batch, and in turn the size of deltaWeights & deltaBiases
 * @return nothing
 */
void initiliseDeltaValues(double**** deltaWeights, double*** deltaBiases, int batchSize) {
	for (int i = 0; i < batchSize; i++) {
		deltaWeights[i] = calloc(networkSize - 1, sizeof(double**));
		deltaBiases[i] = calloc(networkSize - 1, sizeof(double*));

		for (int j = 0; j < networkSize - 1; j++) {
			deltaWeights[i][j] = calloc(networkLayout[j], sizeof(double*));
			deltaBiases[i][j] = calloc(networkLayout[j+1], sizeof(double));

			for (int k = 0; k < networkLayout[j]; k++) 
				deltaWeights[i][j][k] = calloc(networkLayout[j+1], sizeof(double));
		}
	}
}

/**
 * @brief This function frees the memory storing the change of weights and biases, to
 * 	  be called after each batch.
 * @param deltaWeights - stores the change in weights for each of the images in the batch
 * @param deltaBiases - stores the change in biases for each of the imges in the batch
 * @param batchSize - stores the size of the batch, and in turn the size of deltaWeights & deltaBiases
 * @return nothing
 */ 
void destroyDeltaValues(double**** deltaWeights, double*** deltaBiases, int batchSize) {
	for (int i = 0; i < batchSize; i++) {
		for (int j = 0; j < networkSize - 1; j++) {
			for (int k = 0; k < networkLayout[j]; k++) 
				free(deltaWeights[i][j][k]);

			free(deltaWeights[i][j]);
			free(deltaBiases[i][j]);
		}
		free(deltaWeights[i]);
		free(deltaBiases[i]);
	}
	free(deltaWeights);
	free(deltaBiases);
}

/**
 * @brief This function passes an element of the dataset through the network once and records the
 * 	  outputs of each neuron, before and after normilisation.
 * @param input - the element of the dataset to sent through the network
 * @param weights - the weights for each nodes connections
 * @param biases - the biases for each node in the network
 * @param weightedSums - stores the outputs of each node before normilisation, used for back propegation
 * @param outputs - stores the output of each node
 * @return nothin
 */
void frontPropegation(double* input, double*** weights, double** biases, double** weightedSums, double** outputs) {
	outputs[0] = input;
	for (int i = 0; i < networkSize - 1; i++) {
		outputs[i+1] = calloc(networkLayout[i+1], sizeof(double));
		weightedSums[i] = calloc(networkLayout[i+1], sizeof(double));

		for (int j = 0; j < networkLayout[i+1]; j++) {
			outputs[i+1][j] = 0.0;
			for (int k = 0; k < networkLayout[i]; k++) 
				outputs[i+1][j] += outputs[i][k] * weights[i][k][j];

			outputs[i+1][j] += biases[i][j];
			weightedSums[i][j] = outputs[i+1][j];
			outputs[i+1][j] = crelu(outputs[i+1][j]);
		}

		//normilse the output data
		double max = 1;
		for (int j = 0; j < networkLayout[i+1]; j++) 
			if (outputs[i+1][j] > max) 
				max = outputs[i+1][j];

		for (int j = 0; j < networkLayout[i+1]; j++) 
			outputs[i+1][j] /= max;
		
	}
}

/**
 * @brief This function returns the cost of an output
 * @param output - the output of the networks last layer
 * @param expectedValue - the expected output of the network
 * @return The cost of the network, how wrong it is
 */
double getCost(double* output, char expectedValue) {
	int expectedInt = expectedValue - '0';
	double sum = 0.0;
	double temp;

	for (int i = 0; i < networkLayout[networkSize - 1]; i++) {
		temp = output[i] - expectedOutputs[expectedInt][i];
		sum += (temp*temp);
	}

	return sum;
}

/**
 * @brief This function analysies the outputs of the network and compares it to the expected output, records what changes must be made,
 * 	  then sends that information back through the network analysing each node until reaching the input layer
 * @param weights - the weights for the connections between the neurons
 * @param deltaWeights - stores how much the weights should be changed to closer match the expected outputs
 * @param biases - the biases for each node in the network
 * @param deltaBiases - stores how much each bias should be changed to clsoer match the expected outputs
 * @param weightedSums - the outputs for each node before normilisation, taken from the frontpropegation
 * @param outputs - the outputs for each node, taken from the frontporpegation
 * @param expectedVlaue - the expected value for given input
 * @return nothing
 */
void backPropegation(double*** weights, double*** deltaWeights, double** biases, double** deltaBiases, double** weightedSums, double** outputs, char expectedValue) {
	int expectedInt = expectedValue - '0';
	int layer = networkSize - 1;

	//backPropegation from output layer to the previous hidden layer
	for (int i = 0; i < networkLayout[layer]; i++) {
		deltaBiases[layer-1][i] = (outputs[layer][i] - expectedOutputs[expectedInt][i]) * creluPrime(weightedSums[layer-1][i]);
		for (int j = 0; j < networkLayout[layer-1]; j++) 
			deltaWeights[layer-1][j][i] = deltaBiases[layer-1][i] * outputs[layer-1][j];
	}

	//backPropegation from last hidden layer to input layer

	//uses change in bias (not techically but here they're the same value) for the
	//previous layer to compute weights and biases of each consecutive layer
	for (layer--; layer > 0; layer--) {
		for (int i = 0; i < networkLayout[layer]; i++) {
			double temp = 0.0;
			for (int j = 0; j < networkLayout[layer+1]; j++) 
				temp += deltaBiases[layer][j] * weights[layer][i][j];
			deltaBiases[layer-1][i] = temp * creluPrime(weightedSums[layer-1][i]);

			for (int j = 0; j < networkLayout[layer-1]; j++) 
				deltaWeights[layer-1][j][i] = deltaBiases[layer-1][i] * outputs[layer-1][j];
		}
	}

	for (int i = 0; i < networkSize - 1; i++) {
		free(weightedSums[i]);
		free(outputs[i+1]);
	}
}

/**
 * @brief This function shuffles the indexes corresponding to the training dataset
 * @param batchIndexes - the indexes to be shuffled
 * @return nothing
 */
void shuffleBatchIndexes(int* batchIndexes) {
	if (trainingDatasetSize > 1) {
		for (int i = 0; i < trainingDatasetSize - 1; i++) {
			int j = i + rand() / (RAND_MAX / (trainingDatasetSize - i) + 1);
			int temp = batchIndexes[j];
			batchIndexes[j] = batchIndexes[i];
			batchIndexes[i] = temp;
		}
	}
}

/**
 * @bried This function trains the network on one subsuet of the training dataset and then propegates forwards
 * 	  through the network, and then anylises the outputs of each image, recording how much each image should 
 * 	  be changed to closer match the desired result, and then applying those changes.
 * @param batchSize - the number of data points that should be anylised before imporoving the weights and biases
 * @param batchNumber - the current batch, or subset, of the training data.
 * @param bacthIndexes - a randomised array of all of the indexes of the training dataset
 * @param weights - the weights for each of the connection between the neurons 
 * @param biases - the biases for eahc node in the network
 * @param data - the entire training dataset
 * @param labels - the expected outputs for each dataset entry
 * @return nothign
 */
void trainSubset(int batchSize, int batchNumber, int* batchIndexes, double*** weights, double** biases, double** data, char* labels) {
	double**** deltaWeights = calloc(batchSize, sizeof(double***));
	double*** deltaBiases = calloc(batchSize, sizeof(double**));
	initiliseDeltaValues(deltaWeights, deltaBiases, batchSize);

	double** outputs = calloc(networkSize, sizeof(double*));
	double** weightedSums = calloc(networkSize - 1, sizeof(double*));

	//propegate forwards and backwards for each image in the current batch
	for (int i = 0; i < batchSize; i++) {
		frontPropegation(data[batchIndexes[(batchNumber*batchSize)+i]], weights, biases, weightedSums, outputs);
		backPropegation(weights, deltaWeights[i], biases, deltaBiases[i], weightedSums, outputs, labels[batchIndexes[(batchNumber*batchSize)+i]]);
	}

	//average out the weight changes and apply these changes to the nn
	for (int i = 0; i < networkSize - 1; i++) {
		for (int j = 0; j < networkLayout[i]; j++) {
			for (int k = 0; k < networkLayout[i+1]; k++) {
				double averageDeltaW = 0.0;
				for (int ii = 0; ii < batchSize; ii++) 
					averageDeltaW += deltaWeights[ii][i][j][k];

				averageDeltaW /= (double)batchSize;
				weights[i][j][k] -= (learningRate/(double)batchSize) * averageDeltaW;
			}
		}
	}

	//average out the bias changes and apply these changes to the nn
	for (int i = 0; i < networkSize - 1; i++) {
		for (int j = 0; j < networkLayout[i+1]; j++) {
			double averageDeltaB = 0.0;
			for (int k = 0; k < batchSize; k++) 
				averageDeltaB += deltaBiases[k][i][j];

			averageDeltaB /= (double)batchSize;
			biases[i][j] -= (learningRate/(double)batchSize) * averageDeltaB;
		}
	}

	destroyDeltaValues(deltaWeights, deltaBiases, batchSize);
	free(weightedSums);
	free(outputs);
}

/**
 * @brief This function trians the network on a given batch size for a given numebr of epochs
 * @param batchSize - the number of data points that should be anylised before imporoving the weights and biases
 * @param bacthIndexes - a randomised array of all of the indexes of the training dataset
 * @param epochs - the number of times to iterate over the entrie dataset
 * @param weights - the weights for each of the connection between the neurons 
 * @param biases - the biases for eahc node in the network
 * @param data - the entire training dataset
 * @param labels - the expected outputs for each dataset entry
 * @return nothign
 */
void train(int batchSize, int* batchIndexes, int epochs, double*** weights, double** biases, double** data, char* labels) {
	srand(time(NULL));
	double**** deltaWeights;
	double*** deltaBiases;
	double** outputs;

	for (int i = 0; i < epochs; i++) {
		shuffleBatchIndexes(batchIndexes);
		
		for (int i = 0; i < trainingDatasetSize / batchSize; i++) {
			trainSubset(batchSize, i, batchIndexes, weights, biases, data, labels);
		}
		
		printf("epoch %d finished\n", i+1);
	}
}

/**
 * @brief This function tests the neural network on 10000 images that it has never seen before, recording how many it got correct
 * @param weights - the weights for weight nodal connection
 * @param biases - the bias for each node
 * @param data - the testing data taken from the mnist database
 * @prasm labels - the labels for the data
 * @return nothing
 */
void test(double*** weights, double** biases, double** data, char* labels) {
	double averageCost = 0.0;
	int totalCorrect = 0;
	int max, maxIndex;
	double** weightedSums = calloc(networkSize - 1, sizeof(double*));
	double** outputs = calloc(networkSize - 1, sizeof(double*));
	for (int i = 0; i < testingDatasetSize; i++) {
		frontPropegation(data[i], weights, biases, weightedSums, outputs);
		averageCost += getCost(outputs[3], labels[i]);
		max = -100;
		maxIndex = 0;
		for (int j = 0; j < 10; j++) {
			if (outputs[3][j] > outputs[3][maxIndex]) {
				max = outputs[3][j];
				maxIndex = j;
			}
		}

		if (maxIndex == (labels[i] - '0')) 
			totalCorrect++;
	}

	printf("%d/10000\n", totalCorrect);
	free(weightedSums);
	free(outputs);
}

int main(int argc, char* argv[]) {
	if (!(argc == 3 || argc == 4 || argc == 5)) {
		printf("Please enter a command:\n\n");
		printf("Train\t - train <batch size> <num of epochs>\nWill create a new \"Brain\" at data/new_brain and train it for the given number of cyles.\n\n");
		printf("Train\t - train <batch size> <num of epochs> <brain location>\nWill load in the \"Brain\" from the given location and train it for the given time.\n\n");
		printf("Test\t - test <brain location>\nWill test the \"Brain\" at the given location on 10000 images it has never seen before\n");
		return 1;
	}

	//store the dataset data and labels
	double** data;
	char* labels;

	//store the network structure
	double*** weights = calloc(networkSize - 1, sizeof(double**));
	double** biases = calloc(networkSize - 1, sizeof(double*));

	//trains the neural network bases on the user input
	if (strcmp(argv[1], "train") == 0) {
		data = calloc(trainingDatasetSize, sizeof(double*));
		labels = calloc(trainingDatasetSize, sizeof(char));
		loadData(data, labels, "../data/mnist_train.csv");

		argc == 4 ? initialiseNetworkBrain(weights, biases) : loadBrain(weights, biases, argv[4]);

		//store the training data
		int batchSize = atoi(argv[2]);
		int* batchIndexes = calloc(trainingDatasetSize, sizeof(int));
		int epochs = atoi(argv[3]);

		//initilising the batchIndexes to be randomised later
		for (int i = 0; i < trainingDatasetSize; i++) batchIndexes[i] = i;

		train(batchSize, batchIndexes, epochs, weights, biases, data, labels);
		saveBrain(weights, biases, (argc == 4 ? "../data/new_brain" : argv[4]));

		for (int i = 0; i < trainingDatasetSize; i++)
			free(data[i]);
		free(batchIndexes);
	}
	//tests the neural network
	else if (strcmp(argv[1], "test") == 0) {
		data = calloc(testingDatasetSize, sizeof(double*));
		labels = calloc(testingDatasetSize, sizeof(char));
		loadData(data, labels, "../data/mnist_test.csv");

		loadBrain(weights, biases, argv[2]);
		test(weights, biases, data, labels);

		for (int i = 0; i < testingDatasetSize; i++)
			free(data[i]);
	}

	destroyBrainData(weights, biases);
	free(data);
	free(labels);
	free(weights);
	free(biases);

	return 0;
}

#ifndef NETWORKVALUES_H_
#define NETWORKVALUES_H_

#include "NetworkParameters.h"

class NetworkValues
{
public:
	NetworkParameters* networkParameters;

	/*Network Memory State*/
	float longTermMemory[LONG_TERM_MEMORY_ARRAY_SIZE]; // No need for short term memory array, using shortTermMemoryOutputActivation to feed to input

	/*Network Value Sums*/
	float generalHiddenSum[GENERAL_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductHiddenSum[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumHiddenSum[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryHiddenSum[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float outputHiddenSum[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductOutputSum[LONG_TERM_MEMORY_ARRAY_SIZE];
	float longTermMemorySumOutputSum[LONG_TERM_MEMORY_ARRAY_SIZE];
	float shortTermMemoryOutputSum[SHORT_TERM_MEMORY_ARRAY_SIZE];
	float outputOutputSum[OUTPUT_ARRAY_SIZE];

	/*Activation applied to Sums*/
	float generalHiddenActivation[GENERAL_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductHiddenActivation[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumHiddenActivation[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryHiddenActivation[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float outputHiddenActivation[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductOutputActivation[LONG_TERM_MEMORY_ARRAY_SIZE];
	float longTermMemorySumOutputActivation[LONG_TERM_MEMORY_ARRAY_SIZE];
	float shortTermMemoryOutputActivation[SHORT_TERM_MEMORY_ARRAY_SIZE];
	float outputOutputActivation[OUTPUT_ARRAY_SIZE];

	void Assign(NetworkParameters* networkParameter)
	{
		networkParameters = networkParameter;
	}

	void Reset()
	{
		int node;

		for (node = 0; node < LONG_TERM_MEMORY_ARRAY_SIZE; node++)
		{
			longTermMemory[node] = networkParameters->initialLongTermMemory[node];
		}

		for (node = 0; node < SHORT_TERM_MEMORY_ARRAY_SIZE; node++)
		{
			shortTermMemoryOutputActivation[node] = networkParameters->initialShortTermMemory[node];
		}
	}

	void ForwardPropagate(float* input, float* output)
	{
		int hiddenLayer, parentNode, childNode;

		/*General Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for first layer weights in generalHiddenWeights
		{
			generalHiddenSum[0][parentNode] = networkParameters ->generalHiddenBiases[0][parentNode];

			for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				generalHiddenSum[0][parentNode] += longTermMemory[childNode] * networkParameters->longTermMemoryWeights[parentNode][childNode];
			}

			for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				generalHiddenSum[0][parentNode] += shortTermMemoryOutputActivation[childNode] * networkParameters->shortTermMemoryWeights[parentNode][childNode];
			}

			for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
			{
				generalHiddenSum[0][parentNode] += input[childNode] * networkParameters->inputWeights[parentNode][childNode];
			}

			generalHiddenActivation[0][parentNode] = SoftLU(generalHiddenSum[0][parentNode]);
		}

		for (hiddenLayer = 1; hiddenLayer < GENERAL_HIDDEN_LAYERS; hiddenLayer++) // weight layer index offset
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenSum[hiddenLayer][parentNode] = networkParameters->generalHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					generalHiddenSum[hiddenLayer][parentNode] += generalHiddenActivation[hiddenLayer - 1][childNode] * networkParameters->generalHiddenWeights[hiddenLayer - 1][parentNode][childNode];
				}

				generalHiddenActivation[hiddenLayer][parentNode] = SoftLU(generalHiddenSum[hiddenLayer][parentNode]);
			}
		}

		/*Long Term Memory Product Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for the first layer of ltmps
		{
			longTermMemoryProductHiddenSum[0][parentNode] = networkParameters->longTermMemoryProductHiddenBiases[0][parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemoryProductHiddenSum[0][parentNode] += generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode] * networkParameters->longTermMemoryProductHiddenWeights[0][parentNode][childNode];
			}

			longTermMemoryProductHiddenActivation[0][parentNode] = SoftSign(longTermMemoryProductHiddenSum[0][parentNode]);
		}

		for (hiddenLayer = 1; hiddenLayer < LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS; hiddenLayer++) // layer index offset
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemoryProductHiddenSum[hiddenLayer][parentNode] = networkParameters->longTermMemoryProductHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					longTermMemoryProductHiddenSum[hiddenLayer][parentNode] += longTermMemoryProductHiddenActivation[hiddenLayer - 1][childNode] * networkParameters->longTermMemoryProductHiddenWeights[hiddenLayer][parentNode][childNode];
				}

				longTermMemoryProductHiddenActivation[hiddenLayer][parentNode] = SoftSign(longTermMemoryProductHiddenSum[hiddenLayer][parentNode]);
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			longTermMemoryProductOutputSum[parentNode] = networkParameters->longTermMemoryProductOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemoryProductOutputSum[parentNode] += longTermMemoryProductHiddenActivation[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][parentNode] * networkParameters->longTermMemoryProductOutputWeights[parentNode][childNode];
			}

			longTermMemoryProductOutputActivation[parentNode] = SoftSign(longTermMemoryProductOutputSum[parentNode]);
			longTermMemory[parentNode] *= longTermMemoryProductOutputActivation[parentNode];
		}

		/*Long Term Memory Sum Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for the first layer of ltmss
		{
			longTermMemorySumHiddenSum[0][parentNode] = networkParameters->longTermMemorySumHiddenBiases[0][parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemorySumHiddenSum[0][parentNode] += generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode] * networkParameters->longTermMemorySumHiddenWeights[0][parentNode][childNode];
			}

			longTermMemorySumHiddenActivation[0][parentNode] = SoftSign(longTermMemorySumHiddenSum[0][parentNode]);
		}

		for (hiddenLayer = 1; hiddenLayer < LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS; hiddenLayer++) // layer index offset
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemorySumHiddenSum[hiddenLayer][parentNode] = networkParameters->longTermMemorySumHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					longTermMemorySumHiddenSum[hiddenLayer][parentNode] += longTermMemorySumHiddenActivation[hiddenLayer - 1][childNode] * networkParameters->longTermMemorySumHiddenWeights[hiddenLayer][parentNode][childNode];
				}

				longTermMemorySumHiddenActivation[hiddenLayer][parentNode] = SoftSign(longTermMemorySumHiddenSum[hiddenLayer][parentNode]);
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			longTermMemorySumOutputSum[parentNode] = networkParameters->longTermMemorySumOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemorySumOutputSum[parentNode] += longTermMemorySumHiddenActivation[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] * networkParameters->longTermMemorySumOutputWeights[parentNode][childNode];
			}

			longTermMemorySumOutputActivation[parentNode] = SoftSign(longTermMemorySumOutputSum[parentNode]);
			longTermMemory[parentNode] += longTermMemorySumOutputActivation[parentNode];
		}

		/*Short Term Memory Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for the first layer of stmps
		{
			shortTermMemoryHiddenSum[0][parentNode] = networkParameters->shortTermMemoryHiddenBiases[0][parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryHiddenSum[0][parentNode] += generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode] * networkParameters->shortTermMemoryHiddenWeights[0][parentNode][childNode];
			}

			shortTermMemoryHiddenActivation[0][parentNode] = SoftSign(shortTermMemoryHiddenSum[0][parentNode]);
		}

		for (hiddenLayer = 1; hiddenLayer < SHORT_TERM_MEMORY_HIDDEN_LAYERS; hiddenLayer++) // layer index offset
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				shortTermMemoryHiddenSum[hiddenLayer][parentNode] = networkParameters->shortTermMemoryHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					shortTermMemoryHiddenSum[hiddenLayer][parentNode] += shortTermMemoryHiddenActivation[hiddenLayer - 1][childNode] * networkParameters->shortTermMemoryHiddenWeights[hiddenLayer][parentNode][childNode];
				}

				shortTermMemoryHiddenActivation[hiddenLayer][parentNode] = SoftSign(shortTermMemoryHiddenSum[hiddenLayer][parentNode]);
			}
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			shortTermMemoryOutputSum[parentNode] = networkParameters->shortTermMemoryOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryOutputSum[parentNode] += shortTermMemoryHiddenActivation[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] * networkParameters->shortTermMemoryOutputWeights[parentNode][childNode];
			}

			shortTermMemoryOutputActivation[parentNode] = SoftSign(shortTermMemoryOutputSum[parentNode]);
		}

		/*Output Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for the first layer of ops
		{
			outputHiddenSum[0][parentNode] = networkParameters->outputHiddenBiases[0][parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				outputHiddenSum[0][parentNode] += generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode] * networkParameters->outputHiddenWeights[0][parentNode][childNode];
			}

			outputHiddenActivation[0][parentNode] = SoftSign(outputHiddenSum[0][parentNode]);
		}

		for (hiddenLayer = 1; hiddenLayer < OUTPUT_HIDDEN_LAYERS; hiddenLayer++) // layer index offset
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				outputHiddenSum[hiddenLayer][parentNode] = networkParameters->outputHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					outputHiddenSum[hiddenLayer][parentNode] += outputHiddenActivation[hiddenLayer - 1][childNode] * networkParameters->outputHiddenWeights[hiddenLayer][parentNode][childNode];
				}

				outputHiddenActivation[hiddenLayer][parentNode] = SoftSign(outputHiddenSum[hiddenLayer][parentNode]);
			}
		}

		for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
		{
			outputOutputSum[parentNode] = networkParameters->outputOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				outputOutputSum[parentNode] += outputHiddenActivation[OUTPUT_HIDDEN_LAYERS - 1][childNode] * networkParameters->outputOutputWeights[parentNode][childNode];
			}
		}

		Softmax(outputOutputSum, outputOutputActivation);

		for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
		{
			output[parentNode] = outputOutputActivation[parentNode];
		}
	}
};

#endif

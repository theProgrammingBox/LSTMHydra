#ifndef NETWORKGRADIENTS_H_
#define NETWORKGRADIENTS_H_

#include "NetworkValues.h"

class NetworkGradients
{
public:
    NetworkValues* networkValues;

	/*Network Memory State Gradient*/
	float initialLongTermMemoryGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
	float initialShortTermMemoryGradient[SHORT_TERM_MEMORY_ARRAY_SIZE];

	/*Network Weights Gradient*/
	float longTermMemoryWeightsGradient[HIDDEN_LAYER_ARRAY_SIZE][LONG_TERM_MEMORY_ARRAY_SIZE];
	float shortTermMemoryWeightsGradient[HIDDEN_LAYER_ARRAY_SIZE][SHORT_TERM_MEMORY_ARRAY_SIZE];
	float inputWeightsGradient[HIDDEN_LAYER_ARRAY_SIZE][INPUT_ARRAY_SIZE];

	float generalHiddenWeightsGradient[GENERAL_HIDDEN_LAYERS - 1][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE]; // layer index offset

	float longTermMemoryProductHiddenWeightsGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumHiddenWeightsGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryHiddenWeightsGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float outputHiddenWeightsGradient[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductOutputWeightsGradient[LONG_TERM_MEMORY_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumOutputWeightsGradient[LONG_TERM_MEMORY_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryOutputWeightsGradient[SHORT_TERM_MEMORY_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float outputOutputWeightsGradient[OUTPUT_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];

	/*Network Biases Gradient*/
	float generalHiddenBiasesGradient[GENERAL_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductHiddenBiasesGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumHiddenBiasesGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryHiddenBiasesGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float outputHiddenBiasesGradient[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductOutputBiasesGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
	float longTermMemorySumOutputBiasesGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
	float shortTermMemoryOutputBiasesGradient[SHORT_TERM_MEMORY_ARRAY_SIZE];
	float outputOutputBiasesGradient[OUTPUT_ARRAY_SIZE];

    /*Network Memory State Gradient*/
    float longTermMemoryGradient[LONG_TERM_MEMORY_ARRAY_SIZE]; // No need for short term memory array, using shortTermMemoryOutputActivation to feed to input

    /*Network Value Sum Gradients*/
    float generalHiddenSumGradient[GENERAL_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

    float longTermMemoryProductHiddenSumGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
    float longTermMemorySumHiddenSumGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
    float shortTermMemoryHiddenSumGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
    float outputHiddenSumGradient[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

    float longTermMemoryProductOutputSumGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
    float longTermMemorySumOutputSumGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
    float shortTermMemoryOutputSumGradient[SHORT_TERM_MEMORY_ARRAY_SIZE];
    float outputOutputSumGradient[OUTPUT_ARRAY_SIZE];

    /*Activation Gradient to Sums*/
    float generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

    float longTermMemoryProductHiddenActivationGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
    float longTermMemorySumHiddenActivationGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
    float shortTermMemoryHiddenActivationGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
    float outputHiddenActivationGradient[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

    float longTermMemoryProductOutputActivationGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
    float longTermMemorySumOutputActivationGradient[LONG_TERM_MEMORY_ARRAY_SIZE];
    float shortTermMemoryOutputActivationGradient[SHORT_TERM_MEMORY_ARRAY_SIZE];
    float outputOutputActivationGradient[OUTPUT_ARRAY_SIZE];

    void Assign(NetworkValues* networkValue)
    {
       networkValues = networkValue;
    }

    void Reset()
    {
		int hiddenLayer, parentNode, childNode;

        for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			longTermMemoryGradient[parentNode] = 0;
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			shortTermMemoryOutputActivationGradient[parentNode] = 0;
		}

		/*General Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for first layer weights in generalHiddenWeights
		{
			generalHiddenBiasesGradient[0][parentNode] = 0;

			for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				longTermMemoryWeightsGradient[parentNode][childNode] = 0;
			}

			for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryWeightsGradient[parentNode][childNode] = 0;
			}

			for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
			{
				inputWeightsGradient[parentNode][INPUT_ARRAY_SIZE] = 0;
			}
		}

		for (hiddenLayer = 1; hiddenLayer < GENERAL_HIDDEN_LAYERS; hiddenLayer++) // weight layer index offset, LeCun weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenBiasesGradient[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					generalHiddenWeightsGradient[hiddenLayer - 1][parentNode][childNode] = 0;
				}
			}
		}

		/*Long Term Memory Product Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemoryProductHiddenBiasesGradient[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					longTermMemoryProductHiddenWeightsGradient[hiddenLayer][parentNode][childNode] = 0;
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			longTermMemoryProductOutputBiasesGradient[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemoryProductOutputWeightsGradient[parentNode][childNode] = 0;
			}
		}

		/*Long Term Memory Sum Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemorySumHiddenBiasesGradient[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					longTermMemorySumHiddenWeightsGradient[hiddenLayer][parentNode][childNode] = 0;
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			longTermMemorySumOutputBiasesGradient[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemorySumOutputWeightsGradient[parentNode][childNode] = 0;
			}
		}

		/*Short Term Memory Network*/
		for (hiddenLayer = 0; hiddenLayer < SHORT_TERM_MEMORY_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				shortTermMemoryHiddenBiasesGradient[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					shortTermMemoryHiddenWeightsGradient[hiddenLayer][parentNode][childNode] = 0;
				}
			}
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			shortTermMemoryOutputBiasesGradient[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryOutputWeightsGradient[parentNode][childNode] = 0;
			}
		}

		/*Output Network*/
		for (hiddenLayer = 0; hiddenLayer < OUTPUT_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				outputHiddenBiasesGradient[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					outputHiddenWeightsGradient[hiddenLayer][parentNode][childNode] = 0;
				}
			}
		}

		for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			outputOutputBiasesGradient[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				outputOutputWeightsGradient[parentNode][childNode] = 0;
			}
		}
    }

	void UpdateParameters()
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkValues->networkParameters->initialLongTermMemory[parentNode] += longTermMemoryGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkValues->networkParameters->initialShortTermMemory[parentNode] += shortTermMemoryOutputActivationGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;
		}

		/*General Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for first layer weights in generalHiddenWeights
		{
			networkValues->networkParameters->generalHiddenBiases[0][parentNode] += generalHiddenBiasesGradient[0][parentNode] * LEARNING_RATE / BATCH_SIZE;

			for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->longTermMemoryWeights[parentNode][childNode] += longTermMemoryWeightsGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
			}

			for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->shortTermMemoryWeights[parentNode][childNode] += shortTermMemoryWeightsGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
			}

			for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->inputWeights[parentNode][INPUT_ARRAY_SIZE] += inputWeightsGradient[parentNode][INPUT_ARRAY_SIZE] * LEARNING_RATE / BATCH_SIZE;
			}
		}

		for (hiddenLayer = 1; hiddenLayer < GENERAL_HIDDEN_LAYERS; hiddenLayer++) // weight layer index offset, LeCun weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkValues->networkParameters->generalHiddenBiases[hiddenLayer][parentNode] += generalHiddenBiasesGradient[hiddenLayer][parentNode] * LEARNING_RATE / BATCH_SIZE;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkValues->networkParameters->generalHiddenWeights[hiddenLayer - 1][parentNode][childNode] += generalHiddenWeightsGradient[hiddenLayer - 1][parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				}
			}
		}

		/*Long Term Memory Product Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkValues->networkParameters->longTermMemoryProductHiddenBiases[hiddenLayer][parentNode] += longTermMemoryProductHiddenBiasesGradient[hiddenLayer][parentNode] * LEARNING_RATE / BATCH_SIZE;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkValues->networkParameters->longTermMemoryProductHiddenWeights[hiddenLayer][parentNode][childNode] += longTermMemoryProductHiddenWeightsGradient[hiddenLayer][parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			networkValues->networkParameters->longTermMemoryProductOutputBiases[parentNode] += longTermMemoryProductOutputBiasesGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->longTermMemoryProductOutputWeights[parentNode][childNode] += longTermMemoryProductOutputWeightsGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
			}
		}

		/*Long Term Memory Sum Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkValues->networkParameters->longTermMemorySumHiddenBiases[hiddenLayer][parentNode] += longTermMemorySumHiddenBiasesGradient[hiddenLayer][parentNode] * LEARNING_RATE / BATCH_SIZE;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkValues->networkParameters->longTermMemorySumHiddenWeights[hiddenLayer][parentNode][childNode] += longTermMemorySumHiddenWeightsGradient[hiddenLayer][parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			networkValues->networkParameters->longTermMemorySumOutputBiases[parentNode] += longTermMemorySumOutputBiasesGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->longTermMemorySumOutputWeights[parentNode][childNode] += longTermMemorySumOutputWeightsGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
			}
		}

		/*Short Term Memory Network*/
		for (hiddenLayer = 0; hiddenLayer < SHORT_TERM_MEMORY_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkValues->networkParameters->shortTermMemoryHiddenBiases[hiddenLayer][parentNode] += shortTermMemoryHiddenBiasesGradient[hiddenLayer][parentNode] * LEARNING_RATE / BATCH_SIZE;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkValues->networkParameters->shortTermMemoryHiddenWeights[hiddenLayer][parentNode][childNode] += shortTermMemoryHiddenWeightsGradient[hiddenLayer][parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				}
			}
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			networkValues->networkParameters->shortTermMemoryOutputBiases[parentNode] += shortTermMemoryOutputBiasesGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->shortTermMemoryOutputWeights[parentNode][childNode] += shortTermMemoryOutputWeightsGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
			}
		}

		/*Output Network*/
		for (hiddenLayer = 0; hiddenLayer < OUTPUT_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkValues->networkParameters->outputHiddenBiases[hiddenLayer][parentNode] += outputHiddenBiasesGradient[hiddenLayer][parentNode] * LEARNING_RATE / BATCH_SIZE;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkValues->networkParameters->outputHiddenWeights[hiddenLayer][parentNode][childNode] += outputHiddenWeightsGradient[hiddenLayer][parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				}
			}
		}

		for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			networkValues->networkParameters->outputOutputBiases[parentNode] += outputOutputBiasesGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkValues->networkParameters->outputOutputWeights[parentNode][childNode] += outputOutputWeightsGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
			}
		}
	}

    void BackPropagate(float* input, float* output)
    {
        int hiddenLayer, parentNode, childNode;

        /*Output Network*/
        SoftmaxGradient(networkValues->outputOutputSum, outputOutputSumGradient);

        for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
        {
        	outputOutputActivationGradient[parentNode] = CostGradient(networkValues->outputOutputActivation[parentNode], output[parentNode]);
        	outputOutputSumGradient[parentNode] *= outputOutputActivationGradient[parentNode];
			outputOutputBiasesGradient[parentNode] += outputOutputSumGradient[parentNode];
        }

		for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
		{
			outputHiddenActivationGradient[OUTPUT_HIDDEN_LAYERS - 1][childNode] = 0;

			for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
			{
				outputHiddenActivationGradient[OUTPUT_HIDDEN_LAYERS - 1][childNode] += outputOutputSumGradient[parentNode] * networkValues->networkParameters->outputOutputWeights[parentNode][childNode];
				outputOutputWeightsGradient[parentNode][childNode] += outputOutputSumGradient[parentNode] * networkValues->outputHiddenActivation[OUTPUT_HIDDEN_LAYERS - 1][childNode];
			}

			outputHiddenSumGradient[OUTPUT_HIDDEN_LAYERS - 1][childNode] = outputHiddenActivationGradient[OUTPUT_HIDDEN_LAYERS - 1][childNode] * SoftSignGradient(networkValues->outputHiddenSum[OUTPUT_HIDDEN_LAYERS - 1][childNode]);
			outputHiddenBiasesGradient[OUTPUT_HIDDEN_LAYERS - 1][childNode] += outputHiddenSumGradient[OUTPUT_HIDDEN_LAYERS - 1][childNode];
		}

		for (hiddenLayer = OUTPUT_HIDDEN_LAYERS - 2; hiddenLayer >= 0; hiddenLayer--) // layer index offset
		{
			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				outputHiddenActivationGradient[hiddenLayer][childNode] = 0;

				for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
				{
					outputHiddenActivationGradient[hiddenLayer][childNode] += outputHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->networkParameters->outputHiddenWeights[hiddenLayer + 1][parentNode][childNode];
					outputHiddenWeightsGradient[hiddenLayer + 1][parentNode][childNode] += outputHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->outputHiddenActivation[hiddenLayer][childNode];
				}

				outputHiddenSumGradient[hiddenLayer][childNode] = outputHiddenActivationGradient[hiddenLayer][childNode] * SoftSignGradient(networkValues->outputHiddenSum[hiddenLayer][childNode]);
				outputHiddenBiasesGradient[hiddenLayer][childNode] += outputHiddenSumGradient[hiddenLayer][childNode];
			}
		}

		/*Short Term Memory Network*/
        for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
        {
        	shortTermMemoryOutputSumGradient[parentNode] = shortTermMemoryOutputActivationGradient[parentNode] * SoftSignGradient(networkValues->shortTermMemoryOutputSum[parentNode]);
			shortTermMemoryOutputBiasesGradient[parentNode] += shortTermMemoryOutputSumGradient[parentNode];
        }

		for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
		{
			shortTermMemoryHiddenActivationGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode] = 0;

			for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
			{
				shortTermMemoryHiddenActivationGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode] += shortTermMemoryOutputSumGradient[parentNode] * networkValues->networkParameters->shortTermMemoryOutputWeights[parentNode][childNode];
				shortTermMemoryOutputWeightsGradient[parentNode][childNode] += shortTermMemoryOutputSumGradient[parentNode] * networkValues->shortTermMemoryHiddenActivation[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode];
			}

			shortTermMemoryHiddenSumGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode] = shortTermMemoryHiddenActivationGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode] * SoftSignGradient(networkValues->shortTermMemoryHiddenSum[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode]);
			shortTermMemoryHiddenBiasesGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode] += shortTermMemoryHiddenSumGradient[SHORT_TERM_MEMORY_HIDDEN_LAYERS - 1][childNode];
		}

		for (hiddenLayer = SHORT_TERM_MEMORY_HIDDEN_LAYERS - 2; hiddenLayer >= 0; hiddenLayer--) // layer index offset
		{
			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryHiddenActivationGradient[hiddenLayer][childNode] = 0;

				for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
				{
					shortTermMemoryHiddenActivationGradient[hiddenLayer][childNode] += shortTermMemoryHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->networkParameters->shortTermMemoryHiddenWeights[hiddenLayer + 1][parentNode][childNode];
					shortTermMemoryHiddenWeightsGradient[hiddenLayer + 1][parentNode][childNode] += shortTermMemoryHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->shortTermMemoryHiddenActivation[hiddenLayer][childNode];
				}

				shortTermMemoryHiddenSumGradient[hiddenLayer][childNode] = shortTermMemoryHiddenActivationGradient[hiddenLayer][childNode] * SoftSignGradient(networkValues->shortTermMemoryHiddenSum[hiddenLayer][childNode]);
				shortTermMemoryHiddenBiasesGradient[hiddenLayer][childNode] += shortTermMemoryHiddenSumGradient[hiddenLayer][childNode];
			}
		}

		/*Long Term Memory Sum Network*/
        for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
        {
        	longTermMemorySumOutputActivationGradient[parentNode] = longTermMemoryGradient[parentNode];
        	longTermMemorySumOutputSumGradient[parentNode] = longTermMemorySumOutputActivationGradient[parentNode] * SoftSignGradient(networkValues->longTermMemorySumOutputSum[parentNode]);
			longTermMemorySumOutputBiasesGradient[parentNode] += longTermMemorySumOutputSumGradient[parentNode];
        }

		for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
		{
			longTermMemorySumHiddenActivationGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] = 0;

			for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
			{
				longTermMemorySumHiddenActivationGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] += longTermMemorySumOutputSumGradient[parentNode] * networkValues->networkParameters->longTermMemorySumOutputWeights[parentNode][childNode];
				longTermMemorySumOutputWeightsGradient[parentNode][childNode] += longTermMemorySumOutputSumGradient[parentNode] * networkValues->longTermMemorySumHiddenActivation[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode];
			}

			longTermMemorySumHiddenSumGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] = longTermMemorySumHiddenActivationGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] * SoftSignGradient(networkValues->longTermMemorySumHiddenSum[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode]);
			longTermMemorySumHiddenBiasesGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode] += longTermMemorySumHiddenSumGradient[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 1][childNode];
		}

		for (hiddenLayer = LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS - 2; hiddenLayer >= 0; hiddenLayer--) // layer index offset
		{
			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemorySumHiddenActivationGradient[hiddenLayer][childNode] = 0;

				for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
				{
					longTermMemorySumHiddenActivationGradient[hiddenLayer][childNode] += longTermMemorySumHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->networkParameters->longTermMemorySumHiddenWeights[hiddenLayer + 1][parentNode][childNode];
					longTermMemorySumHiddenWeightsGradient[hiddenLayer + 1][parentNode][childNode] += longTermMemorySumHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->longTermMemorySumHiddenActivation[hiddenLayer][childNode];
				}

				longTermMemorySumHiddenSumGradient[hiddenLayer][childNode] = longTermMemorySumHiddenActivationGradient[hiddenLayer][childNode] * SoftSignGradient(networkValues->longTermMemorySumHiddenSum[hiddenLayer][childNode]);
				longTermMemorySumHiddenBiasesGradient[hiddenLayer][childNode] += longTermMemorySumHiddenSumGradient[hiddenLayer][childNode];
			}
		}

		/*Long Term Memory Product Network*/
        for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
        {
        	longTermMemoryProductOutputActivationGradient[parentNode] = longTermMemoryGradient[parentNode] * networkValues->longTermMemory[parentNode];
        	longTermMemoryProductOutputSumGradient[parentNode] = longTermMemoryProductOutputActivationGradient[parentNode] * SoftSignGradient(networkValues->longTermMemoryProductOutputSum[parentNode]);
			longTermMemoryProductOutputBiasesGradient[parentNode] += longTermMemoryProductOutputSumGradient[parentNode];
        }

		for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
		{
			longTermMemoryProductHiddenActivationGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode] = 0;

			for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
			{
				longTermMemoryProductHiddenActivationGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode] += longTermMemoryProductOutputSumGradient[parentNode] * networkValues->networkParameters->longTermMemoryProductOutputWeights[parentNode][childNode];
				longTermMemoryProductOutputWeightsGradient[parentNode][childNode] += longTermMemoryProductOutputSumGradient[parentNode] * networkValues->longTermMemoryProductHiddenActivation[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode];
			}

			longTermMemoryProductHiddenSumGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode] = longTermMemoryProductHiddenActivationGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode] * SoftSignGradient(networkValues->longTermMemoryProductHiddenSum[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode]);
			longTermMemoryProductHiddenBiasesGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode] += longTermMemoryProductHiddenSumGradient[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 1][childNode];
		}

		for (hiddenLayer = LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS - 2; hiddenLayer >= 0; hiddenLayer--) // layer index offset
		{
			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemoryProductHiddenActivationGradient[hiddenLayer][childNode] = 0;

				for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
				{
					longTermMemoryProductHiddenActivationGradient[hiddenLayer][childNode] += longTermMemoryProductHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->networkParameters->longTermMemoryProductHiddenWeights[hiddenLayer + 1][parentNode][childNode];
					longTermMemoryProductHiddenWeightsGradient[hiddenLayer + 1][parentNode][childNode] += longTermMemoryProductHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->longTermMemoryProductHiddenActivation[hiddenLayer][childNode];
				}

				longTermMemoryProductHiddenSumGradient[hiddenLayer][childNode] = longTermMemoryProductHiddenActivationGradient[hiddenLayer][childNode] * SoftSignGradient(networkValues->longTermMemoryProductHiddenSum[hiddenLayer][childNode]);
				longTermMemoryProductHiddenBiasesGradient[hiddenLayer][childNode] += longTermMemoryProductHiddenSumGradient[hiddenLayer][childNode];
			}
		}

		/*General Network*/
        for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
        {
        	generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] = 0;

			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] += outputHiddenSumGradient[0][parentNode] * networkValues->networkParameters->outputHiddenWeights[0][parentNode][childNode];
				outputHiddenWeightsGradient[0][parentNode][childNode] += outputHiddenSumGradient[0][parentNode] * networkValues->generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode];
			}

			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] += shortTermMemoryHiddenSumGradient[0][parentNode] * networkValues->networkParameters->shortTermMemoryHiddenWeights[0][parentNode][childNode];
				shortTermMemoryHiddenWeightsGradient[0][parentNode][childNode] += shortTermMemoryHiddenSumGradient[0][parentNode] * networkValues->generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode];
			}

			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] += longTermMemorySumHiddenSumGradient[0][parentNode] * networkValues->networkParameters->longTermMemorySumHiddenWeights[0][parentNode][childNode];
				longTermMemorySumHiddenWeightsGradient[0][parentNode][childNode] += longTermMemorySumHiddenSumGradient[0][parentNode] * networkValues->generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode];
			}

			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] += longTermMemoryProductHiddenSumGradient[0][parentNode] * networkValues->networkParameters->longTermMemoryProductHiddenWeights[0][parentNode][childNode];
				longTermMemoryProductHiddenWeightsGradient[0][parentNode][childNode] += longTermMemoryProductHiddenSumGradient[0][parentNode] * networkValues->generalHiddenActivation[GENERAL_HIDDEN_LAYERS - 1][childNode];
			}

			generalHiddenSumGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] = generalHiddenActivationGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] * SoftLUGradient(networkValues->generalHiddenSum[GENERAL_HIDDEN_LAYERS - 1][childNode]);
			generalHiddenBiasesGradient[GENERAL_HIDDEN_LAYERS - 1][childNode] += generalHiddenSumGradient[GENERAL_HIDDEN_LAYERS - 1][childNode];
        }

		for (hiddenLayer = GENERAL_HIDDEN_LAYERS - 2; hiddenLayer >= 0; hiddenLayer--) // layer index offset
		{
			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				generalHiddenActivationGradient[hiddenLayer][childNode] = 0;

				for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
				{
					generalHiddenActivationGradient[hiddenLayer][childNode] += generalHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->networkParameters->generalHiddenWeights[hiddenLayer][parentNode][childNode];
					generalHiddenWeightsGradient[hiddenLayer][parentNode][childNode] += generalHiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->generalHiddenActivation[hiddenLayer][childNode];
				}

				generalHiddenSumGradient[hiddenLayer][childNode] = generalHiddenActivationGradient[hiddenLayer][childNode] * SoftLUGradient(networkValues->generalHiddenSum[hiddenLayer][childNode]);
				generalHiddenBiasesGradient[hiddenLayer][childNode] += generalHiddenSumGradient[hiddenLayer][childNode];
			}
		}

		for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemoryGradient[childNode] += generalHiddenSumGradient[0][parentNode] * networkValues->networkParameters->longTermMemoryWeights[parentNode][childNode];
				longTermMemoryWeightsGradient[parentNode][childNode] += generalHiddenSumGradient[0][parentNode] * networkValues->longTermMemory[childNode];
			}
		}

		for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
		{
			shortTermMemoryOutputActivationGradient[childNode] = 0;

			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				shortTermMemoryOutputActivationGradient[childNode] += generalHiddenSumGradient[0][parentNode] * networkValues->networkParameters->shortTermMemoryWeights[parentNode][childNode];
				shortTermMemoryWeightsGradient[parentNode][childNode] += generalHiddenSumGradient[0][parentNode] * networkValues->shortTermMemoryOutputActivation[childNode];
			}
		}

		for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				inputWeightsGradient[parentNode][childNode] += generalHiddenSumGradient[0][parentNode] * input[childNode];
			}
		}
    }
};

#endif

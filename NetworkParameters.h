#ifndef NETWORKPARAMETERS_H_
#define NETWORKPARAMETERS_H_

#include "Configurations.h"

class NetworkParameters
{
public:
	/*Network Memory State*/
	float initialLongTermMemory[LONG_TERM_MEMORY_ARRAY_SIZE];
	float initialShortTermMemory[SHORT_TERM_MEMORY_ARRAY_SIZE];

	/*Network Weights*/
	float longTermMemoryWeights[HIDDEN_LAYER_ARRAY_SIZE][LONG_TERM_MEMORY_ARRAY_SIZE];
	float shortTermMemoryWeights[HIDDEN_LAYER_ARRAY_SIZE][SHORT_TERM_MEMORY_ARRAY_SIZE];
	float inputWeights[HIDDEN_LAYER_ARRAY_SIZE][INPUT_ARRAY_SIZE];

	float generalHiddenWeights[GENERAL_HIDDEN_LAYERS - 1][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE]; // layer index offset

	float longTermMemoryProductHiddenWeights[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumHiddenWeights[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryHiddenWeights[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float outputHiddenWeights[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductOutputWeights[LONG_TERM_MEMORY_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumOutputWeights[LONG_TERM_MEMORY_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryOutputWeights[SHORT_TERM_MEMORY_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];
	float outputOutputWeights[OUTPUT_ARRAY_SIZE][HIDDEN_LAYER_ARRAY_SIZE];

	/*Network Biases*/
	float generalHiddenBiases[GENERAL_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductHiddenBiases[LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float longTermMemorySumHiddenBiases[LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float shortTermMemoryHiddenBiases[SHORT_TERM_MEMORY_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];
	float outputHiddenBiases[OUTPUT_HIDDEN_LAYERS][HIDDEN_LAYER_ARRAY_SIZE];

	float longTermMemoryProductOutputBiases[LONG_TERM_MEMORY_ARRAY_SIZE];
	float longTermMemorySumOutputBiases[LONG_TERM_MEMORY_ARRAY_SIZE];
	float shortTermMemoryOutputBiases[SHORT_TERM_MEMORY_ARRAY_SIZE];
	float outputOutputBiases[OUTPUT_ARRAY_SIZE];

	void Initialize()
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			initialLongTermMemory[parentNode] = 0;
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			initialShortTermMemory[parentNode] = 0;
		}

		/*General Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++) // accounts for first layer weights in generalHiddenWeights,  // LeCun weight initialization
		{
			generalHiddenBiases[0][parentNode] = 0;

			for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				longTermMemoryWeights[parentNode][childNode] = NormalRandom(0, 1.0 / LONG_TERM_MEMORY_ARRAY_SIZE);
			}

			for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryWeights[parentNode][childNode] = NormalRandom(0, 1.0 / SHORT_TERM_MEMORY_ARRAY_SIZE);
			}

			for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
			{
				inputWeights[parentNode][INPUT_ARRAY_SIZE] = NormalRandom(0, 1.0 / INPUT_ARRAY_SIZE);
			}
		}

		for (hiddenLayer = 1; hiddenLayer < GENERAL_HIDDEN_LAYERS; hiddenLayer++) // weight layer index offset, LeCun weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				generalHiddenBiases[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					generalHiddenWeights[hiddenLayer - 1][parentNode][childNode] = NormalRandom(0, 1.0 / HIDDEN_LAYER_ARRAY_SIZE);
				}
			}
		}

		/*Long Term Memory Product Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemoryProductHiddenBiases[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					longTermMemoryProductHiddenWeights[hiddenLayer][parentNode][childNode] = NormalRandom(0, 1.0 / (HIDDEN_LAYER_ARRAY_SIZE));
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			longTermMemoryProductOutputBiases[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemoryProductOutputWeights[parentNode][childNode] = NormalRandom(0, 2.0 / (HIDDEN_LAYER_ARRAY_SIZE + LONG_TERM_MEMORY_ARRAY_SIZE));
			}
		}

		/*Long Term Memory Sum Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				longTermMemorySumHiddenBiases[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					longTermMemorySumHiddenWeights[hiddenLayer][parentNode][childNode] = NormalRandom(0, 1.0 / (HIDDEN_LAYER_ARRAY_SIZE));
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			longTermMemorySumOutputBiases[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				longTermMemorySumOutputWeights[parentNode][childNode] = NormalRandom(0, 2.0 / (HIDDEN_LAYER_ARRAY_SIZE + LONG_TERM_MEMORY_ARRAY_SIZE));
			}
		}

		/*Short Term Memory Network*/
		for (hiddenLayer = 0; hiddenLayer < SHORT_TERM_MEMORY_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				shortTermMemoryHiddenBiases[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					shortTermMemoryHiddenWeights[hiddenLayer][parentNode][childNode] = NormalRandom(0, 1.0 / HIDDEN_LAYER_ARRAY_SIZE);
				}
			}
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			shortTermMemoryOutputBiases[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				shortTermMemoryOutputWeights[parentNode][childNode] = NormalRandom(0, 2.0 / (HIDDEN_LAYER_ARRAY_SIZE + SHORT_TERM_MEMORY_ARRAY_SIZE));
			}
		}

		/*Output Network*/
		for (hiddenLayer = 0; hiddenLayer < OUTPUT_HIDDEN_LAYERS; hiddenLayer++) // Glorot weight initialization
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				outputHiddenBiases[hiddenLayer][parentNode] = 0;

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					outputHiddenWeights[hiddenLayer][parentNode][childNode] = NormalRandom(0, 1.0 / HIDDEN_LAYER_ARRAY_SIZE);
				}
			}
		}

		for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++) // Glorot weight initialization
		{
			outputOutputBiases[parentNode] = 0;

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				outputOutputWeights[parentNode][childNode] = NormalRandom(0, 2.0 / (HIDDEN_LAYER_ARRAY_SIZE + OUTPUT_ARRAY_SIZE));
			}
		}
	}

	void ExportNetwork(string fileName)
	{
		ofstream networkOut(fileName);

		int hiddenLayer, parentNode, childNode;

		networkOut << LONG_TERM_MEMORY_ARRAY_SIZE << ' ';
		networkOut << SHORT_TERM_MEMORY_ARRAY_SIZE << ' ';
		networkOut << INPUT_ARRAY_SIZE << ' ';
		networkOut << HIDDEN_LAYER_ARRAY_SIZE << ' ';
		networkOut << OUTPUT_ARRAY_SIZE << ' ';

		networkOut << GENERAL_HIDDEN_LAYERS << ' ';
		networkOut << LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS << ' ';
		networkOut << LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS << ' ';
		networkOut << OUTPUT_HIDDEN_LAYERS << ' ';
		networkOut << SHORT_TERM_MEMORY_HIDDEN_LAYERS << ' ';

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkOut << initialLongTermMemory[parentNode] << ' ';
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkOut << initialShortTermMemory[parentNode] << ' ';
		}

		/*General Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
		{
			networkOut << generalHiddenBiases[0][parentNode] << ' ';

			for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				networkOut << longTermMemoryWeights[parentNode][childNode] << ' ';
			}

			for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				networkOut << shortTermMemoryWeights[parentNode][childNode] << ' ';
			}

			for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
			{
				networkOut << inputWeights[parentNode][INPUT_ARRAY_SIZE] << ' ';
			}
		}

		for (hiddenLayer = 1; hiddenLayer < GENERAL_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkOut << generalHiddenBiases[hiddenLayer][parentNode] << ' ';

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkOut << generalHiddenWeights[hiddenLayer - 1][parentNode][childNode] << ' ';
				}
			}
		}

		/*Long Term Memory Product Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkOut << longTermMemoryProductHiddenBiases[hiddenLayer][parentNode] << ' ';

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkOut << longTermMemoryProductHiddenWeights[hiddenLayer][parentNode][childNode] << ' ';
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkOut << longTermMemoryProductOutputBiases[parentNode] << ' ';

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkOut << longTermMemoryProductOutputWeights[parentNode][childNode] << ' ';
			}
		}

		/*Long Term Memory Sum Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkOut << longTermMemorySumHiddenBiases[hiddenLayer][parentNode] << ' ';

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkOut << longTermMemorySumHiddenWeights[hiddenLayer][parentNode][childNode] << ' ';
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkOut << longTermMemorySumOutputBiases[parentNode] << ' ';

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkOut << longTermMemorySumOutputWeights[parentNode][childNode] << ' ';
			}
		}

		/*Short Term Memory Network*/
		for (hiddenLayer = 0; hiddenLayer < SHORT_TERM_MEMORY_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkOut << shortTermMemoryHiddenBiases[hiddenLayer][parentNode] << ' ';

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkOut << shortTermMemoryHiddenWeights[hiddenLayer][parentNode][childNode] << ' ';
				}
			}
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkOut << shortTermMemoryOutputBiases[parentNode] << ' ';

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkOut << shortTermMemoryOutputWeights[parentNode][childNode] << ' ';
			}
		}

		/*Output Network*/
		for (hiddenLayer = 0; hiddenLayer < OUTPUT_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkOut << outputHiddenBiases[hiddenLayer][parentNode] << ' ';

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkOut << outputHiddenWeights[hiddenLayer][parentNode][childNode] << ' ';
				}
			}
		}

		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
		{
			networkOut << outputOutputBiases[parentNode] << ' ';

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkOut << outputOutputWeights[parentNode][childNode] << ' ';
			}
		}

		networkOut.close();
	}

	void ImportNetwork(string fileName)
	{
		ifstream networkIn(fileName);

		int hiddenLayer, parentNode, childNode;
		int longTermMemoryArraySize, shortTermMemoryArraySize, inputArraySize, hiddenArraySize, outputArraySize;
		int generalHiddenLayers, longTermMemoryProductHiddenLayers, longTermMemorySumHiddenLayers, outputHiddenLayers, shortTermMemoryHiddenLayers;

		networkIn >> longTermMemoryArraySize;
		if (longTermMemoryArraySize != LONG_TERM_MEMORY_ARRAY_SIZE)
		{
			cout << "Mismatching LONG_TERM_MEMORY_ARRAY_SIZE. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> shortTermMemoryArraySize;
		if (shortTermMemoryArraySize != SHORT_TERM_MEMORY_ARRAY_SIZE)
		{
			cout << "Mismatching SHORT_TERM_MEMORY_ARRAY_SIZE. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> inputArraySize;
		if (inputArraySize != INPUT_ARRAY_SIZE)
		{
			cout << "Mismatching INPUT_ARRAY_SIZE. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> hiddenArraySize;
		if (hiddenArraySize != HIDDEN_LAYER_ARRAY_SIZE)
		{
			cout << "Mismatching HIDDEN_LAYER_ARRAY_SIZE. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> outputArraySize;
		if (outputArraySize != OUTPUT_ARRAY_SIZE)
		{
			cout << "Mismatching OUTPUT_ARRAY_SIZE. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> generalHiddenLayers;
		if (generalHiddenLayers != GENERAL_HIDDEN_LAYERS)
		{
			cout << "Mismatching GENERAL_HIDDEN_LAYERS. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> longTermMemoryProductHiddenLayers;
		if (longTermMemoryProductHiddenLayers != LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS)
		{
			cout << "Mismatching LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> longTermMemorySumHiddenLayers;
		if (longTermMemorySumHiddenLayers != LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS)
		{
			cout << "Mismatching LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> outputHiddenLayers;
		if (outputHiddenLayers != OUTPUT_HIDDEN_LAYERS)
		{
			cout << "Mismatching OUTPUT_HIDDEN_LAYERS. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}

		networkIn >> shortTermMemoryHiddenLayers;
		if (shortTermMemoryHiddenLayers != SHORT_TERM_MEMORY_HIDDEN_LAYERS)
		{
			cout << "Mismatching SHORT_TERM_MEMORY_HIDDEN_LAYERS. Press enter to reinitialize the network. ";
			cin.ignore(1000, '\n');
			Initialize();
			return;
		}


		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkIn >> initialLongTermMemory[parentNode];
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkIn >> initialShortTermMemory[parentNode];
		}

		/*General Network*/
		for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
		{
			networkIn >> generalHiddenBiases[0][parentNode];

			for (childNode = 0; childNode < LONG_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				networkIn >> longTermMemoryWeights[parentNode][childNode];
			}

			for (childNode = 0; childNode < SHORT_TERM_MEMORY_ARRAY_SIZE; childNode++)
			{
				networkIn >> shortTermMemoryWeights[parentNode][childNode];
			}

			for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
			{
				networkIn >> inputWeights[parentNode][INPUT_ARRAY_SIZE];
			}
		}

		for (hiddenLayer = 1; hiddenLayer < GENERAL_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkIn >> generalHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkIn >> generalHiddenWeights[hiddenLayer - 1][parentNode][childNode];
				}
			}
		}

		/*Long Term Memory Product Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkIn >> longTermMemoryProductHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkIn >> longTermMemoryProductHiddenWeights[hiddenLayer][parentNode][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkIn >> longTermMemoryProductOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkIn >> longTermMemoryProductOutputWeights[parentNode][childNode];
			}
		}

		/*Long Term Memory Sum Network*/
		for (hiddenLayer = 0; hiddenLayer < LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkIn >> longTermMemorySumHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkIn >> longTermMemorySumHiddenWeights[hiddenLayer][parentNode][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < LONG_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkIn >> longTermMemorySumOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkIn >> longTermMemorySumOutputWeights[parentNode][childNode];
			}
		}

		/*Short Term Memory Network*/
		for (hiddenLayer = 0; hiddenLayer < SHORT_TERM_MEMORY_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkIn >> shortTermMemoryHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkIn >> shortTermMemoryHiddenWeights[hiddenLayer][parentNode][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < SHORT_TERM_MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkIn >> shortTermMemoryOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkIn >> shortTermMemoryOutputWeights[parentNode][childNode];
			}
		}

		/*Output Network*/
		for (hiddenLayer = 0; hiddenLayer < OUTPUT_HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_LAYER_ARRAY_SIZE; parentNode++)
			{
				networkIn >> outputHiddenBiases[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
				{
					networkIn >> outputHiddenWeights[hiddenLayer][parentNode][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
		{
			networkIn >> outputOutputBiases[parentNode];

			for (childNode = 0; childNode < HIDDEN_LAYER_ARRAY_SIZE; childNode++)
			{
				networkIn >> outputOutputWeights[parentNode][childNode];
			}
		}

		networkIn.close();
	}
};

#endif

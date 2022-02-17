#ifndef NETWORKTRAINER_H_
#define NETWORKTRAINER_H_

#include "Environment.h"
#include "NetworkGradients.h"

class NetworkTrainer
{
public:
	Environment environment;
	NetworkParameters networkParameters;
	NetworkValues networkValuesArray[SEQUENCE_SIZE];
	NetworkGradients networkGradients;

	float input[SEQUENCE_SIZE][INPUT_ARRAY_SIZE];
	float output[SEQUENCE_SIZE][OUTPUT_ARRAY_SIZE];
	float networkOutput[SEQUENCE_SIZE][OUTPUT_ARRAY_SIZE];

	void Initialize()
	{
		networkParameters.Initialize();
	}

	void Train(int iterations)
	{
		int iteration, batch, sequence;
		int node;
		float AverageError;

		AverageError = 0;
		for (iteration = 0; iteration < iterations; iteration++)
		{
			networkGradients.Reset();

			for (batch = 0; batch < BATCH_SIZE; batch++)
			{
				NetworkValues networkValues;

				environment.Initialize();
				networkValues.Assign(&networkParameters);
				networkValues.Reset();

				for (sequence = 0; sequence < SEQUENCE_SIZE; sequence++)
				{
					environment.GetInput(input[sequence]);
					environment.GetOutput(output[sequence]);
					networkValues.ForwardPropagate(input[sequence], networkOutput[sequence]);
					networkValuesArray[sequence] = networkValues;
					environment.ForwardPropagate();

					for (node = 0; node < OUTPUT_ARRAY_SIZE; node++)
					{
						AverageError += Cost(networkOutput[sequence][node], output[sequence][node]);
					}

//					cout << "input:\n";
//					for (node = 0; node < OUTPUT_ARRAY_SIZE; node++)
//					{
//						cout << input[sequence][node] << " ";
//					}
//					cout << endl;
//
//					cout << "networkOutput:\n";
//					for (node = 0; node < OUTPUT_ARRAY_SIZE; node++)
//					{
//						cout << networkOutput[sequence][node] << " ";
//					}
//					cout << endl;
//
//					cout << "output:\n";
//					for (node = 0; node < OUTPUT_ARRAY_SIZE; node++)
//					{
//						cout << output[sequence][node] << " ";
//					}
//					cout << endl;

//					for (node = 0; node < OUTPUT_ARRAY_SIZE; node++)
//					{
//						cout << Cost(networkOutput[sequence][node], output[sequence][node]) << " ";
//					}
//					cout << endl;
//
//					for (node = 0; node < OUTPUT_ARRAY_SIZE; node++)
//					{
//						cout << CostGradient(networkOutput[sequence][node], output[sequence][node]) << " ";
//					}
//					cout << endl;
//					cout << endl;
				}

				for (sequence = SEQUENCE_SIZE - 1; sequence >= 0; sequence--)
				{
					networkGradients.Assign(&networkValuesArray[sequence]);
					networkGradients.BackPropagate(input[sequence], output[sequence]);
				}
			}

			networkGradients.UpdateParameters();
		}
		cout << "AVERAGE ERROR: " << AverageError / (iterations * BATCH_SIZE * SEQUENCE_SIZE * OUTPUT_ARRAY_SIZE) << endl;
	}

	void SaveNetwork(string fileName)
	{
		networkParameters.ExportNetwork(fileName);
	}

	void LoadNetwork(string fileName)
	{
		networkParameters.ImportNetwork(fileName);
	}
};



#endif

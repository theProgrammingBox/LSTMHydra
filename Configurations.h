#ifndef CONFIGURATIONS_H_
#define CONFIGURATIONS_H_

#include <iostream>
#include <string>
#include <fstream>
#include "Random.h"

using namespace std;

/*Training Parameters*/
const int BATCH_SIZE = 8;
const int SEQUENCE_SIZE = 10;
const float LEARNING_RATE = 0.0001;

/*Network Parameters*/
const int LONG_TERM_MEMORY_ARRAY_SIZE = 2;
const int SHORT_TERM_MEMORY_ARRAY_SIZE = 2;
const int INPUT_ARRAY_SIZE = 2;
const int HIDDEN_LAYER_ARRAY_SIZE = 4;
const int OUTPUT_ARRAY_SIZE = 2;

const int GENERAL_HIDDEN_LAYERS = 2;					// Minimum of 2
const int LONG_TERM_MEMORY_PRODUCT_HIDDEN_LAYERS = 2;	// Minimum of 2
const int LONG_TERM_MEMORY_SUM_HIDDEN_LAYERS = 2;		// Minimum of 2
const int OUTPUT_HIDDEN_LAYERS = 2;						// Minimum of 2
const int SHORT_TERM_MEMORY_HIDDEN_LAYERS = 2;			// Minimum of 2

const float SoftSign(float x)
{
	return x / (1 + abs(x));
}

const float SoftSignGradient(float x)
{
	return 1 / pow(1 + abs(x), 2);
}

const float SoftLU(float x)
{
	return x * (1 + SoftSign(x));
}

const float SoftLUGradient(float x)
{
	return 1 + SoftSign(x) + x * SoftSignGradient(x);
}

const void Softmax(float* input, float* output)
{
	int parentNode;
	float largestValue;
	float total;

	largestValue = input[OUTPUT_ARRAY_SIZE];

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		if (largestValue < input[parentNode])
		{
			largestValue = input[parentNode];
		}
	}

	total = 0;

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] = exp(input[parentNode] - largestValue);
		total += output[parentNode];
	}

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] /= total;
	}
}

const void SoftmaxGradient(float* input, float* output)
{
	int parentNode, childNode;
	float largestValue;
	float total;
	float numerator[INPUT_ARRAY_SIZE]{};

	largestValue = input[OUTPUT_ARRAY_SIZE];

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		if (largestValue < input[parentNode])
		{
			largestValue = input[parentNode];
		}
	}

	total = 0;

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] = input[parentNode] - largestValue;
		total += exp(output[parentNode]);
	}

	total *= total;

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		for (childNode = 0; childNode < OUTPUT_ARRAY_SIZE; childNode++)
		{
			if (parentNode != childNode)
			{
				numerator[parentNode] += exp(output[parentNode] + output[childNode]);
			}
		}
	}

	for (parentNode = 0; parentNode < OUTPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] = numerator[parentNode] / total;
	}
}

const float Cost(float output, float expected) // Expected must be 1 or 0
{
	if (expected == 1)
	{
		if (output < 0.0000000001)
		{
			return 10;
		}
		return -log(output);
	}
	else if (expected == 0)
	{
		if (output > 0.9999999999)
		{
			return 10;
		}
		return -log(1 - output);
	}
}

const float CostGradient(float output, float expected) // Expected must be 1 or 0
{

	if (expected == 1)
	{
		if (output < 0.01)
		{
			return 100;
		}
		return 1 / output;
	}
	else if (expected == 0)
	{
		if (output > 0.99)
		{
			return -100;
		}
		return 1 / (output - 1);
	}
}

#endif

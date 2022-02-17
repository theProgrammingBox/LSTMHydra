#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "Configurations.h"

class Environment
{
public:
	const int memoryLag = 3;
	bool memory[3];
	int memoryIndex;

	void Initialize()
	{
		memoryIndex = 0;

		for (int i = 0; i < memoryLag; i++)
		{
			memory[i] = 0;
		}

		ForwardPropagate();
	}

	void ForwardPropagate()
	{
		bool randVal = (UIntRandom1() ^ UIntRandom2()) % 2;
		memoryIndex = memoryIndex + 1 == memoryLag? 0 : memoryIndex + 1;

		memory[memoryIndex] = randVal;
	}

	void GetInput(float* input)
	{
		for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
		{
			input[i] = memory[memoryIndex] == i;
		}
	}

	void GetOutput(float* output)
	{
		for (int i = 0; i < OUTPUT_ARRAY_SIZE; i++)
		{
			output[i] = memory[memoryIndex + 1 == memoryLag? 0 : memoryIndex + 1] == i;
		}
	}
};

#endif

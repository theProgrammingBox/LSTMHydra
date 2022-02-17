#include "NetworkTrainer.h"

int main()
{
	NetworkTrainer networkTrainer;

	networkTrainer.LoadNetwork("Network.txt");
	networkTrainer.Initialize();

	while(true)
	{
		networkTrainer.Train(1000);
		networkTrainer.SaveNetwork("Network.txt");
	}

	return 0;
}

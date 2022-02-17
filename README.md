# LSTMHydra
## A LSTM Varient That Implements A Joined Neural Network
### The LSTMHydra model Is Made of:
1. A General Network. The activation function is SoftLU. The input is the long term memory line, output of the Short Term Memory Network, and input array
2. A Long Term Memory Product Network. The activation function is SoftSign. The input is the output of the General Network. The output is multiplied to the long term memory line.
3. A Long Term Memory Sum Network. The activation function is SoftSign. The input is the output of the General Network. The output is added to the long term memory line.
4. A Short Term Memory Sum Network. The activation function is SoftSign. The input is the output of the General Network. The output is fed to the General Network in the next iteration.
5. An Output Network. The activation function is SoftSign. The input is the output of the General Network. The output is the output of the entire model.
### Other Important Details
1. The current model's output is Softmaxed so I am using a logarithmic cost function.
2. The current environment the model is being trained to remember a number that was specified 2 iterations ago, for 10 iterations.
3. The weights are initialized to optimised normal distributed random numbers based on the activation function of the network.
4. The biases are initialized to 0.

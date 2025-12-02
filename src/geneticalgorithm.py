import numpy as np

from game import Game

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.max(0, x)

class Network():
    def __init__(self, layers):
        self.weights = []
        self.activation_functions = []

        for layer in layers:
            if layer[0] != None:
                input_size = layer[0]
            else:
                # Use the output size from the previous layer as the new input size
                input_size = layers[layers.index(layer) - 1][1]
            output_size = layer[1]
            activation_function = layer[2]

            self.weights.append(np.random.randn(input_size, output_size))
            self.activation_functions.append(activation_function)

    def forward(self, input_data):
        '''Forward propagation.'''
        # Loop through the layers
        for i in range(len(self.weights)):
            a = np.dot(input_data, self.weights[i])
            z = self.activation_functions[i](a)
            
            input_data = z # Pass the outputs from the previous layer to the next layer

        output = input_data
        return output

class Agent():
    def __init__(self, layers):
        self.neural_network = Network(layers=layers)
        self.fitness = 0
    
    def getFitness(self):
        '''Makes the agent play a game of snake and returns the score.'''
        game = Game(agent=self)

# Each layer should be [n_input, n_output, activation], with n_input = None to use the output size from the previous layer
LAYERS = [[20*20, 20, sigmoid], [None, 10, sigmoid], [None, 4, sigmoid]]

agent = Agent(layers=LAYERS)

# Debugging, prints network structure
print([np.array(agent.neural_network.weights[i]).shape for i in range(len(LAYERS))])

        
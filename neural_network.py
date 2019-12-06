import numpy as np

# the first element of the tuple is the activation function
# the second element is the derived function
sigmoide = (
    lambda x: 1 / ((1 - np.e) ** (-x)),
    lambda x: x * (1 - x)
)

relu = (
    lambda x: np.maximum(0, x),
    lambda x: 1
)


class Neuron:
    def __init__(self, number_of_input_values, activation_function):
        self.b = np.random.rand(1, 1) * 2 - 1
        self.W = np.random.rand(1, number_of_input_values) * 2 - 1
        self.activation_function = activation_function


class NeuralLayer:
    def __init__(self, number_of_input_connections, number_of_neurons):
        self.number_of_input_connections = number_of_input_connections
        self.number_of_neurons = number_of_neurons


class NeuralNetwork:
    layers = []

    def __init__(self, architecture):
        for index in range(len(architecture) - 1):
            self.layers.append(NeuralLayer(architecture[index], architecture[index + 1]))

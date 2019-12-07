import numpy as np

# the first element of the tuple is the activation function
# the second element is the derived function
## Activation functions
sigmoid = (lambda x: 1 / (1 + np.e ** (-x)),
           lambda x: x * (1 - x))

relu = (lambda x: np.maximum(0, x),
        lambda x: 1)

## Cost functions
cost_function = (
    lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
    lambda Yp, Yr: (Yp - Yr)
)


class NeuralLayer:
    def __init__(self, number_of_input_connections, number_of_neurons, activation_function):
        self.activation_function = activation_function
        self.number_of_neurons = number_of_neurons
        self.number_of_input_connections = number_of_input_connections

        self.b = np.random.rand(1, number_of_neurons) * - 1
        self.W = np.random.rand(number_of_input_connections, number_of_neurons) * 2 - 1


class NeuralNetwork:
    layers = []

    def __init__(self, architecture, activation_function):
        for index in range(len(architecture) - 1):
            self.layers.append(NeuralLayer(architecture[index], architecture[index + 1], activation_function))


def train(neural_network, X, Y, cost_function, learning_rate=0.5, train_mode=True):
    out = [(None, X)]

    # Forward pass
    for layer in neural_network.layers:
        print(out[-1][1])
        print(layer.W)

        z = out[-1][1] @ layer.W + layer.b
        a = layer.activation_function[0](z)

        out.append((z, a))


nn = NeuralNetwork([2, 4, 6], sigmoid)

import numpy as np

from sklearn.datasets import make_circles

n = 5
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

train(nn, X, None, None)

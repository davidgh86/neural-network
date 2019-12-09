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


def train(neural_network, X_entry_values, Y_expected_value, function_cost, learning_rate=0.5, train_mode=True):
    out = [(None, X_entry_values)]

    # Forward pass
    for layer in neural_network.layers:

        z = out[-1][1] @ layer.W + layer.b
        a = layer.activation_function[0](z)

        out.append((z, a))

    # Backward pass
    if train_mode:

        deltas = []

        for network_layer_index in reversed(range(0, len(neural_network.layers))):

            z = out[network_layer_index + 1][0]
            a = out[network_layer_index + 1][1]

            if network_layer_index == (len(neural_network.layers) - 1):
                deltas.insert(0, function_cost[1](a, Y_expected_value) * neural_network.layers[network_layer_index]
                              .activation_function[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_network.layers[network_layer_index]
                              .activation_function[1](a))

            _W = neural_network.layers[network_layer_index].W

            # Gradient descent
            neural_network.layers[network_layer_index].b = \
                neural_network.layers[network_layer_index].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
            neural_network.layers[network_layer_index].W = \
                neural_network.layers[network_layer_index].W - out[network_layer_index][1].T @ deltas[0] * learning_rate

    return out[-1][1]



nn = NeuralNetwork([2, 4, 6], sigmoid)

import numpy as np

from sklearn.datasets import make_circles

n = 5
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

train(nn, X, Y, cost_function)

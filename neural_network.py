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

        self.b = np.random.rand(1, number_of_neurons) * - 1
        self.W = np.random.rand(number_of_input_connections, number_of_neurons) * 2 - 1


class NeuralNetwork:
    layers = []

    def __init__(self, architecture, activation_function=sigmoid, function_cost=cost_function):
        self.function_cost = function_cost
        for index in range(len(architecture) - 1):
            self.layers.append(NeuralLayer(architecture[index], architecture[index + 1], activation_function))

    def train(self, X_entry_values, Y_expected_value, learning_rate=0.5, train_mode=True):
        out = [(None, X_entry_values)]

        # Forward pass
        for layer in self.layers:
            z = out[-1][1] @ layer.W + layer.b
            a = layer.activation_function[0](z)

            out.append((z, a))

        # Backward pass
        if train_mode:

            deltas = []

            for network_layer_index in reversed(range(0, len(self.layers))):

                a = out[network_layer_index + 1][1]

                if network_layer_index == (len(self.layers) - 1):
                    deltas.insert(0, self.function_cost[1](a, Y_expected_value) * self.layers[network_layer_index]
                                  .activation_function[1](a))
                else:
                    deltas.insert(0, deltas[0] @ _W.T * self.layers[network_layer_index]
                                  .activation_function[1](a))

                _W = self.layers[network_layer_index].W

                # Gradient descent
                self.layers[network_layer_index].b = \
                    self.layers[network_layer_index].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
                self.layers[network_layer_index].W = \
                    self.layers[network_layer_index].W - out[network_layer_index][1].T @ deltas[0] * learning_rate

        return out[-1][1]


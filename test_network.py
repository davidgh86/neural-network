import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
import neural_network

from sklearn.datasets import make_circles

# Create dataset

n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

# The data must be represented as an two-dimensional array
Y = Y[:, np.newaxis]

# plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
# plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
# plt.axis("equal")
# plt.show()

nn = neural_network.NeuralNetwork([p, 4, 8, 1])

# registering error
# loss = []

for i in range(3000):
    pY = nn.train(X, Y, learning_rate=0.5)
    print(neural_network.cost_function[0](pY, Y))

# Error should approximate to 0

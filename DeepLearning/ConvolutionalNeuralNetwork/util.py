import numpy as np


# helper functions
###############################################################
def cross_entropy(batch_size, output, expected_output):
    return (-1 / batch_size) * np.sum(expected_output * np.log(output) + (1 - expected_output) * np.log(1 - output))


def normalize(z):
    return relu(z)
    # return sigmoid(z)


def normalize_prime(z):
    return relu_prime(z)
    # return sigmoid_prime(z)


def relu(z):
    return np.where(z > 0, z, 0.0)


def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def loss(desired, final):
    return 0.5 * np.sum(desired - final) ** 2

import numpy as np
from Layer import Layer
from util import normalize


class FullyConnectedLayer(Layer):
    """
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    """
    def __init__(self, input_shape, num_output):
        super(FullyConnectedLayer, self).__init__(input_shape, num_output)
        self.depth, self.height_in, self.width_in = input_shape
        self.num_output = num_output

        self.weights = np.random.randn(self.num_output, self.depth, self.height_in, self.width_in)
        self.biases = np.random.rand(self.num_output, 1)

    def feed_forward(self, a):
        """
        propagates forward through the FC layer to the final output layer
        """
        # roll out the dimensions
        self.weights = self.weights.reshape((self.num_output, self.depth * self.height_in * self.width_in))
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        # this is shape of (num_outputs, 1)
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = normalize(self.z_values)
        self.weights = self.weights.reshape((self.num_output, self.depth, self.height_in, self.width_in))
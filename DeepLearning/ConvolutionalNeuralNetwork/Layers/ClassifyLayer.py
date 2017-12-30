import numpy as np
import Layer
from util import sigmoid


class ClassifyLayer(Layer):
    def __init__(self, num_inputs, num_classes):
        super(ClassifyLayer, self).__init__(num_inputs, num_classes)
        num_inputs, col = num_inputs
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_classes, num_inputs)
        self.biases = np.random.randn(self.num_classes, 1)

    def classify(self, x):
        self.z_values = np.dot(self.weights, x) + self.biases
        self.output = sigmoid(self.z_values)

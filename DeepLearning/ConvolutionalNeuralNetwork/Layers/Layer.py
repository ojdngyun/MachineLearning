import numpy as np


class Layer(object):
    def __init__(self, input_shape, num_output):
        self.output = np.ones((num_output, 1))
        self.z_values = np.ones((num_output, 1))

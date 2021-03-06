import numpy as np
from util import normalize


class ConvolutionalLayer(object):

    def __init__(self, input_shape, filter_size, stride, num_filters, padding=0):
        self.depth, self.height_in, self.width_in = input_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters

        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size)
        self.biases = np.random.randn(self.num_filters, 1)

        # TODO generalize output dimension with stride value if needed
        # current formula only works for unit strides
        self.output_dim1 = (self.height_in - self.filter_size) + (2 * self.padding) + 1  # num of rows
        self.output_dim2 = (self.width_in - self.filter_size) + (2 * self.padding) + 1  # num of cols

        self.z_values = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))

    def convolve(self, image_data):
        """
        convolve input image with filter
        :param image_data: image
        :return: sigmoid activation matrix after convolution
        """

        # roll out activation
        self.z_values = self.z_values.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1 * self.output_dim2))

        act_length1d = self.output.shape[1]

        for j in range(self.num_filters):
            slide = 0
            row = 0

            for i in range(act_length1d):  # loop till the output array is filled up -> 1 dimensional (600)
                # activation -> loop through each convolutional block horizontally
                # image data is a tuple (depth, 28, 28)
                self.z_values[j][i] = np.sum(
                    image_data[:, row:self.filter_size + row, slide:self.filter_size + slide] * self.weights[j]
                ) + self.biases[j]
                # normalization using sigmoid -> output 0...1.0
                self.output[j][i] = normalize(self.z_values[j][i])
                slide += self.stride

                if (self.filter_size + slide) - self.stride >= self.width_in:  # wrap indices at the end of each row
                    slide = 0
                    row += self.stride

        self.z_values = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1, self.output_dim2))

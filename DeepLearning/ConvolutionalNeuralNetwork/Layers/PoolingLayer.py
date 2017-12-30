import numpy as np
import Layer


class PoolingLayer(Layer):

    def __init__(self, input_shape, poolsize=(2, 2)):
        """
        :param input_shape: width_in and height_in are the dimensions of the input image
        :param poolsize: is treated as a tuple of filter and stride -> it should work with overlapping pooling
        """
        self.depth, self.height_in, self.width_in = input_shape
        self.poolsize = poolsize
        self.height_out = (self.height_in - self.poolsize[0]) / self.poolsize[1] + 1
        self.width_out = (self.width_in - self.poolsize[0])  # num of output neurons

        self.output = np.empty((self.depth, self.height_out, self.width_out))
        self.max_indices = np.empty((self.depth, self.height_out, self.width_out, 2))

    def pool(self, input_image):
        self.pool_length1d = self.height_out * self.width_out

        self.output = self.output.reshape((self.depth, self.pool_length1d))
        self.max_indices = self.max_indices.reshape((self.depth, self.pool_length1d, 2))

        # for each filter map
        for j in range(self.depth):
            row = 0
            slide = 0
            for i in range(self.pool_length1d):
                to_pool = input_image[j][row:self.poolsize[0] + row, slide:self.poolsize[0] + slide]

                self.output[j][i] = np.amax(to_pool)  # calcuate the max activation
                index = zip(*np.where(np.max(to_pool) == to_pool))
                print('PoolingLayer', index, type(index), len(index))
                if len(index) > 1:
                    index = [index[0]]
                index = index[0][0] + row, index[0][1] + slide
                self.max_indices[j][i] = index

                slide += self.poolsize[1]

                # modify this if stride != filter for poolsize
                if slide >= self.width_in:
                    slide = 0
                    row += self.poolsize[1]

        self.output = self.output.reshape((self.depth, self.height_out, self.width_out))
        self.max_indices = self.max_indices.reshape((self.depth, self.height_out, self.width_out, 2))


from ConvolutionalLayer import ConvolutionalLayer
from FullyConnectedLayer import FullyConnectedLayer
from PoolingLayer import PoolingLayer
from ClassifyLayer import ClassifyLayer
from util import *
from Backprop import *
import matplotlib.pyplot as plt
import numpy as np
import time
import random


class Model(object):
    FULLY_CONNECTED_LAYER = 'fc_layer'
    FINAL_LAYER = 'final_layer'
    CONVOLUTIONAL_LAYER = 'conv_layer'
    POOLING_LAYER = 'pool_layer'

    layer_type_map = {
        FULLY_CONNECTED_LAYER: FullyConnectedLayer,
        FINAL_LAYER: ClassifyLayer,
        CONVOLUTIONAL_LAYER: ConvolutionalLayer,
        POOLING_LAYER: PoolingLayer
    }

    def __init__(self, input_shape, layer_config):
        """
        :param layer_config: list of dicts, outer key is
        Valid Layer Types:
        Convolutional Layer: shape of input, filter_size, stride, padding, num_filters
        Pooling Layer: shape of input(depth, height_in, width_in), poolsize
        Fully Connected Layer: shape_of_input, num_output, classify = True/False, num_classes (if classify True)
        Gradient Descent: training data, batch_size, eta, num_epochs, lambda, test_data
        """
        self.input_shape = input_shape
        self._initialize_layers(layer_config)
        self.layer_weight_shapes = [l.weights.shape for l in self.layers if not isinstance(l, PoolingLayer)]
        self.layer_biases_shapes = [l.biases.shape for l in self.layers if not isinstance(l, PoolingLayer)]

    def _initialize_layers(self, layer_config):
        """
        :param layer_config: Sets the network's layer attribute to be a list of layers
        (classes from layer_type_map)
        """
        layers = []
        input_shape = self.input_shape
        for layer_spec in layer_config:
            # handle the spec format: { 'type': {kwargs}}
            layer_class = self.layer_type_map[layer_spec.keys()[0]]
            layer_kwargs = layer_spec.values()[0]
            layer = layer_class(input_shape, **layer_kwargs)
            # output shape for the current layer = input shape for the next layer
            input_shape = layer.output.shape
            layers.append(layer)
        self.layers = layers

    def _get_layer_transition(self, inner_ix, outer_ix):
        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        # either input to FC or pool to FC -> going from 3d matrix to 1d
        if (
                    (inner_ix < 0 or isinstance(inner, PoolingLayer)) and
                    isinstance(outer, FullyConnectedLayer)
        ):
            return '3d_to_1d'

        # going from 3d to 3d matrix -> either input to conv or conv to conv
        if (
                    (inner_ix < 0 or isinstance(inner, ConvolutionalLayer)) and
                    isinstance(outer, ConvolutionalLayer)
        ):
            return 'to_conv'

        if (
                    isinstance(inner, FullyConnectedLayer) and
                    (isinstance(outer, ClassifyLayer) or isinstance(outer, FullyConnectedLayer))
        ):
            return '1d_to_1d'

        if (
                    isinstance(inner, ConvolutionalLayer) and
                    isinstance(outer, PoolingLayer)
        ):
            return 'conv_to_pool'

        raise NotImplementedError

    def feed_forward(self, image):
        """
        :param image: image to processed through network
        :return: returns final classification of the image
        """
        prev_activation = image

        # forward pass
        for layer in self.layers:
            input_to_feed = prev_activation

            if isinstance(layer, FullyConnectedLayer):
                # z values are huge, while the fc_output is tiny! negative values are cut off and set to zero
                layer.feed_forward(input_to_feed)
            elif isinstance(layer, ConvolutionalLayer):
                layer.convolve(input_to_feed)
                # for i in range(layer.output.shape[0]):
                #     plt.imsave('images/cat_conv%d.jpg'%i, layer.output[i])
                # for i in range(layer.weights.shape[0]):
                #     plt.imsave('images/filter_conv%s.jpg'%i, layer.weights[i].reshape((5, 5)))
            elif isinstance(layer, PoolingLayer):
                layer.pool(input_to_feed)
                # for i in range(layer.output.shape[0]):
                #     plt.imsave('images/pool_pic%s.jpg'%i, layer.output[i])
            elif isinstance(layer, ClassifyLayer):
                layer.classify(input_to_feed)
            else:
                raise NotImplementedError

            # output from current layer to be used to the next one
            prev_activation = layer.output

        final_activation = prev_activation
        return final_activation

    def backprop(self, image, label):
        """
        :param image: image data
        :param label: label
        :return:
        """
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        # set first params on the final layer
        final_output = self.layers[-1].output
        last_delta = (final_output - label) * normalize(self.layers[-1].z_values)
        last_weights = None
        final = True

        num_layers = len(self.layers)

        for l in range(num_layers - 1, -1, -1):
            # the outer layer is closer to classification
            # the inner layer is closer to input
            inner_layer_ix = l - 1
            if (l - 1) < 0:
                inner_layer_ix = 0
            outer_layer_ix = l

            layer = self.layers[outer_layer_ix]
            activation = self.layers[inner_layer_ix].output if inner_layer_ix >= 0 else image

            transition = self._get_layer_transition(
                inner_layer_ix, outer_layer_ix
            )

            # input fc = pool fc
            # fc to fc = fc to final
            # conv to conv -> input to conv
            # conv to pool -> unique

            if transition == '1d_to_1d':  # final to fc, fc to fc
                db, dw, last_delta = backprop_1d_to_1d(
                    delta=last_delta,
                    prev_weights=last_weights,
                    prev_activations=activation,
                    z_vals=layer.z_values,
                    final=final
                )
                final = False
            elif transition == '3d_to_1d':
                if l == 0:
                    activation = image
                # calc delta on the first final layer
                db, dw, last_delta = backprop_1d_to_3d(
                    delta=last_delta,
                    prev_weights=last_weights,  # shape (10, 100) this is the weights from the next layer
                    prev_activations=activation,  # (28, 28)
                    z_vals=layer.z_values  # (100, 1)
                )
                # layer.weights = layer.weights.reshape((layer.num_output, layer.depth, layer.height_in, layer.width_in))
            elif transition == 'conv_to_pool':  # pool to conv layer
                # no update for dw, db => only backprops the error
                last_delta = backprop_pool_to_conv(
                    delta=last_delta,
                    prev_weights=last_weights,
                    input_from_conv=activation,
                    max_indices=layer.max_indices,
                    poolsize=layer.poolsize,
                    pool_output=layer.output
                )
            elif transition == 'to_conv':  # conv to conv layer
                # weights passed in are the ones between conv to conv
                # update the weights and biases
                activation = image
                last_weights = layer.weights,
                db, dw = backprop_to_conv(
                    delta=last_delta,
                    weight_filters=layer.weights,
                    stride=layer.stride,
                    input_to_conv=activation,
                    prev_z_vals=layer.z_values
                )
            else:
                pass

            if transition != 'conv_to_pool':
                # print 'nablasb, db,nabldw, dw, DELTA', nabla_b[inner_layer_ix].shape, db.shape, nabla_w[inner_layer_ix].shape, dw.shape, last_delta.shape
                nabla_b[inner_layer_ix], nabla_w[inner_layer_ix] = db, dw
                last_weights = layer.weights

        return self.layers[-1].output, nabla_b, nabla_w

    def update_mini_batch(self, batch, eta):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        batch_size = len(batch)

        error = 0
        for image, label in batch:
            image = image.reshape((1, 28, 28))
            _ = self.feed_forward(image)
            final_res, delta_b, delta_w = self.backprop(image, label)

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

            error = loss(label, final_res)

        num = 0
        weight_index = []
        for layer in self.layers:
            if not isinstance(layer, PoolingLayer):
                weight_index.append(num)
            num += 1

        for ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
            layer = self.layers[weight_index[ix]]
            layer.weights -= eta * layer_nabla_w / batch_size
            layer.biases -= eta * layer_nabla_b / batch_size

        return error

    def validate(self, data):
        data = [(im.reshape((1, 28, 28)), y) for im, y in data]
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y, in data]
        return sum(int(x == y) for x, y in test_results)

    def train(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data=None):
        training_size = len(training_data)

        mean_error = []
        correct_res = []

        for epoch in range(num_epochs):
            print('starting training, epoch: ', epoch)
            start = time.time()
            random.shuffle(training_data)
            # dividing the training set into batches
            batches = [training_data[k: k + batch_size] for k in range(0, training_size, batch_size)]
            losses = 0

            for index, batch in enumerate(batches):
                print('currentIndex: ', index, 'batch count: ', len(batches), 'current epoch: ', epoch)
                current_loss = self.update_mini_batch(batch, eta)
                losses += current_loss
            mean_error.append(round(losses / batch_size, 2))
            print(mean_error)

            if test_data:
                print('validation step')
                res = self.validate(test_data)
                correct_res.append(res)
                test_size = len(test_data)
                print("Epoch {0}: {1}/{2}, success rate: {3}%".format(
                    epoch, res, test_size, (float(res) / float(test_size)) * 100
                ))
                timer = time.time() - start
                print("Estimated time for this epoch: ", timer)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(correct_res)
        plt.show()

import mnist_loader
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
import numpy as np
from Model import Model

ETA = 1.5
EPOCHS = 30
INPUT_SHAPE = 28 * 28
BATCH_SIZE = 10
LMBDA = 0.1

def digit_count(data_list):
    dictionary = {}
    for value in data_list:
        dictionary[value[1]] = dictionary.get(value[1], 0) + 1
    return dictionary

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print 'training_data size: ', len(training_data)
print 'validation_data size: ', len(validation_data), digit_count(validation_data)
print 'test_data size: ', len(test_data), digit_count(test_data)
print '\n'

x, y = training_data[0][0].shape
input_shape = (1, x, y)
print 'shape of input data: ', input_shape

cnn = Model(input_shape,
            layer_config = [
                {Model.CONVOLUTIONAL_LAYER: {
                    'filter_size': 5,
                    'stride': 1,
                    'num_filters': 20
                }},
                {Model.POOLING_LAYER: {
                    'poolsize': (2, 2)
                }},
                {Model.FULLY_CONNECTED_LAYER: {
                    'num_output': 30
                }},
                {Model.FINAL_LAYER: {
                    'num_classes': 10
                }}
            ])

training_size = 0.002
# train = False
train = True

if train:
    cnn.train(
        training_data[:int(round(len(training_data) * training_size))],
        BATCH_SIZE,
        ETA,
        EPOCHS,
        LMBDA,
        test_data = test_data[:int(round(len(test_data) * training_size))]
    )
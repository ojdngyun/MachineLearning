"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import cPickle
import gzip

import numpy as np


def load_data():
    """
    :return: mnist datas as a tuple containing the training data,
    the validation data and the test data
    The training data is returned as a tuple with two entries.
    The first entry contains the actual training images. This is a
    numpy ndarray with 50,000 entries. Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image
    The second entry in the training data tuple is a numpy ndarray
    containing 50,000 entries. Those entries are just the digit values
    (0...9) for the corresponding images contained in the first entry
    of the tuple.
    The validation and test data are similar, except each contains only
    10,000 images.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def vectorize_result(j):
    """
    :return: a 10-dimensional unit vector with a 1.0 in jth
    position and zeroes elsewhere. This is used to convert
    a digit (0...9) into a corresponding desired output from
    neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    """
    :return: a tuple containing training, validation and test data
    The training data is a list containing 50,000 2-tuples (x,y).
    x is 784-dimensional numpy.ndarray containing the input image.
    y is a 10-dimensional numpy.ndarray representing the unit vector
    corresponding to the correct digit for x.
    Validation and test data are lists containing 10,000 2-tuples (x, y).
    In each case x is a 784-dimensional numpy.ndarray containing the input
    image and y is the corresponding classification.
    The training and validation/test data are of different format for
    convenience in the neural network.
    """
    tr_data, va_data, te_data = load_data()
    training_inputs = [np.reshape(x, (28, 28)) for x in tr_data[0]]
    training_results = [vectorize_result(y) for y in tr_data[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (28, 28)) for x in va_data[0]]
    validation_data = zip(validation_inputs, va_data[1])

    test_inputs = [np.reshape(x, (28, 28)) for x in te_data[0]]
    test_data = zip(test_inputs, te_data[1])

    return training_data, validation_data, test_data


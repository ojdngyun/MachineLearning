import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import urllib.request
from random import *
from mnist_util import *

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.cmap'] = 'Greys'

import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

train_images, train_labels, test_images, test_labels = load_mnist_file('MNIST_data')
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)


is_train = True
# is_train = False


def test():
    print('asdf')


def test_data():
    random_index = randint(0, len(test_labels))
    image = test_images[random_index]
    label = test_labels[random_index]

    print(label)
    plt.imshow(image.reshape((28, 28)))
    plt.show()


def train_model():
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    print(mnist.train.images.shape)


def train_linear_model():
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # using cross entropy as loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # initializing tensorflow variable and session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # iterative training step
    batch_size = 100
    for i in range(1000):
        batch_index = (i % int(len(train_labels) / batch_size)) * batch_size
        batch_xss = train_images[batch_index:batch_index + batch_size]
        batch_yss = train_labels[batch_index:batch_index + batch_size]
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # x and y_ are the tensorflow variable defined above
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))
    print(sess.run(accuracy, feed_dict={x: train_images, y_: train_labels}))

    image = test_images[11]
    label = test_labels[11]
    print('right: ', label)
    plt.imshow(image.reshape(28, 28))

    print('predicted: ', sess.run(y, feed_dict={x: [image]}))
    plt.show()

if is_train:
    train_linear_model()
    # train_model()
else:
    test()

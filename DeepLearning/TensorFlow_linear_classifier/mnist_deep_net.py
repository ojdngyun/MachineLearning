from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def train_single_layer_softmax_regression():
    mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    sess = tf.InteractiveSession()

    # placeholder -> variable to put inputs for the computational graph
    # x -> flatten input image (28, 28)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # y_ -> image label in one_hot form e.g: 4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # tensorflow variable -> values in tensorflow computational graph
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # initializing and setting initial values to the variables
    sess.run(tf.global_variables_initializer())

    # defining regression model
    y = tf.matmul(x, W) + b

    # defining loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # 1d array of boolean
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # boolean casted to integers [true, false, true, true] -> [1, 0, 1, 1]
    # reduce_mean calculates the mean and the result is 0.75
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval())


def weight_variable(shape):
    """
    the weights are initialize with a small amount of noise for
    symmetry breaking and to prevent 0 gradients
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    since ReLU neurons are being, it's better to have a slightly
    positive bias
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train_multilayer_conv_net():
    mnist = input_data.read_data_sets('mnist_data', one_hot=True)

    # placeholder -> variable to put inputs for the computational graph
    # x -> flatten input image (28, 28)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # y_ -> image label in one_hot form e.g: 4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # 1st layer: convolution + max_pooling #####################
    # convolution: 32 filters each of size 5x5
    # the first 2 dimensions are the patch size, 3rd is
    # the number on input channels and the last one is
    # the number of output channels (filters)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convolve x_image with weight tensor -> add bias -> apply ReLU -> max pool
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # the max pool operation will half the image to 14x14
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd layer is also a convolutional layer ###################
    # convolution: 64 filters each of 5x5
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # image size halved again 7x7
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer ###################################
    # fully connected layer with 1024 neurons to process entire image
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout layer ############################################
    # This is to reduce overfitting
    # A placeholder for the probability that a neuron's output is kept
    # during a dropout. This allows toggle dropout during training and testing
    # tf.nn.dropout operation automatically handles scaling neuron outputs in
    # addition to masking them, so dropout just works without any additional scaling
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer ############################################
    # final layer with weights and bias
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def test():
    # mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
    print('asdfasdf')


train_multilayer_conv_net()
# train_single_layer_softmax_regression()
# test()

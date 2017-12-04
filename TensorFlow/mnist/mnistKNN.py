import numpy as np
import tensorflow as tf

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data

# store the mnist in /tmp/data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.test.next_batch(200)

training_digits_pl = tf.placeholder("float", [None, 784])

test_digit_pl = tf.placeholder("float", [784])

# nearest neighbor calculation using L1 distance
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

# prediction: get min distance index (nearest neighbor)
pred = tf.argmin(distance, 0)

accuracy = 0.

# initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(test_digits)):
        # get nearest neighbor
        nn_index = sess.run(pred, feed_dict={training_digits_pl: training_digits, test_digit_pl: test_digits[i, :]})

        # get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction: ", np.argmax(training_labels[nn_index]), "True Label: ", np.argmax(test_labels[i]))

        # calculate accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits)

    print("Done!")
    print("Accuracy: ", accuracy)

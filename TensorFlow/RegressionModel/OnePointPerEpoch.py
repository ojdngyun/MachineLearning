import numpy as np
import tensorflow as tf
from sklearn import datasets, linear_model

from returns_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()
print(xData, yData)

# set up a linear model to represent this
googModel = linear_model.LinearRegression()

googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

# find the coeff and intercept of this linear model
print(googModel.coef_)
print(googModel.intercept_)

###############################################################
# Simple regression - one point per epoch using tensorflow

# model linear regression y = Wx + b
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

# placeholder to feed in the returns, returns have many rows,
# just one column
x = tf.placeholder(tf.float32, [None, 1])

Wx = tf.matmul(x, W)

y = Wx + b

# placeholder to hold the y-labels, also returns
y_ = tf.placeholder(tf.float32, [None, 1])

# y_: predicted value
# y : actual value
cost = tf.reduce_mean(tf.square(y_ - y))

train_step_constant = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# set up a method to perform the actual training. Allow us to
# modify the optimizer used and also the number of steps
# in the training
def trainWithOnePointEpoch(steps, train_step):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(steps):

            # extract one training point
            xs = np.array([[xData[i % len(xData)]]])
            ys = np.array([[yData[i % len(yData)]]])

            feed = {x: xs, y_: ys}

            sess.run(train_step, feed_dict=feed)

            # print result to the screen for every 1000 iterations
            if (i + 1) % 1000 == 0:
                print('After %d iteration:' % i)

                print('W: %f' % sess.run(W))
                print('b: %f' % sess.run(b))

                print('cost: %f' % sess.run(cost, feed_dict=feed))


trainWithOnePointEpoch(10000, train_step_constant)

import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plot
import os

filename = "./dandelion.jpg"

image = mp_img.imread(filename)

print('image shape: ', image.shape)
print('image array: ', image)
plot.imshow(image)
plot.show()

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # original axis indexes 0, 1, 2
    # swapping the first and second axis (height and width values)
    # transpose = tf.transpose(x, perm=[1, 0, 2])
    transpose = tf.image.transpose_image(x)

    result = sess.run(transpose)

    print('Transposed image shape: ', result.shape)
    plot.imshow(result)
    plot.show()

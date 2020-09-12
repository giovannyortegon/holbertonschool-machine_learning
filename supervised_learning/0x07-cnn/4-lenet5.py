#!/usr/bin/env python3
""" LeNet-5 """
import tensorflow as tf


def lenet5(x, y):
    """ lenet5 - builds a modified version of the LeNet-5 architecture

    Args:
        x   is a tf.placeholder of shape (m, 28, 28, 1) containing
            the input images for the network.

            m   is the number of images.

        y   is a tf.placeholder of shape (m, 10) containing the one-hot
            labels for the network.

    Return:
        softmax, optimizer, loss, accuracy
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    c1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                          padding='same', activation=tf.nn.relu,
                          kernel_initializer=init)(x)

    s2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c3 = tf.layers.Conv2D(filters=16, kernel_size=5,
                          padding='valid', activation=tf.nn.relu,
                          kernel_initializer=init)(s2)

    s4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)
    f = tf.layers.Flatten()(s4)

    c5 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                         kernel_initializer=init)(f)

    f6 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                         kernel_initializer=init)(f6)

    output = tf.layers.Dense(units=10, kernel_initializer=init)(f6)

    softmax = tf.nn.softmax(output)
    loss = tf.losses.softmax_cross_entropy(y, output)
    pred = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return output, optimizer, loss, accuracy

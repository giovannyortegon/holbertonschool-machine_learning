#!/usr/bin/env python3
""" mini-batch gradient descent """
import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ train_mini_batch - trains a loaded neural network model
                           using mini-batch gradient descent:

    Args:
        X_train is a numpy.ndarray of shape (m, 784) containing
                the training data
            m is the number of data points
            784 is the number of input features

        Y_train is a one-hot numpy.ndarray of shape (m, 10) containing
                the training labels.
            10 is the number of classes the model should classify

        X_valid is a numpy.ndarray of shape (m, 784) containing
                the validation data
        Y_valid is a one-hot numpy.ndarray of shape (m, 10)
                containing the validation labels

        batch_size is the number of data points in a batch
        epochs is the number of times the training should pass
                through the whole dataset

        load_path is the path from which to load the model
        save_path is the path to where the model should be saved after training

    Returns:
        the path where the model was saved
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(session, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]

        if m % batch_size == 0:
            complete = 1
            num_mini_batches = int(m / batch_size)
        else:
            complete = 0
            num_mini_batches = int(m / batch_size) + 1

        for i in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}

            train_cost, train_accuracy = session.run(
                [loss, accuracy], feed_train
            )
            valid_cost, valid_accuracy = session.run(
                [loss, accuracy], feed_valid
            )

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

                for k in range(num_mini_batches):
                    if complete == 0 and k == num_mini_batches - 1:
                        start = k * batch_size
                        mini_batch_X = shuffled_X[start::]
                        mini_batch_Y = shuffled_Y[start::]
                    else:
                        start = k * batch_size
                        end = (k * batch_size) + batch_size
                        mini_batch_X = shuffled_X[start: end]
                        mini_batch_Y = shuffled_Y[start: end]

                    feed_mini_batch = {x: mini_batch_X, y: mini_batch_Y}
                    session.run(train_op, feed_mini_batch)

                    step = k + 1
                    if step % 100 == 0 and k != 0:
                        step_cost, step_accuracy = session.run(
                            [loss, accuracy], feed_mini_batch
                        )
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

    return saver.save(session, save_path)

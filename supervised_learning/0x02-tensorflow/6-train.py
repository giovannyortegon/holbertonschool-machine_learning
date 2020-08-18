#!/usr/bin/env python3
""" train """
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """ train - builds, trains and saves a neural network

    Args:
        X_train is a numpy.ndarray containing the training input data.
        Y_train is a numpy.ndarray containing the training labels.
        X_valid is a numpy.ndarray containing the validation input data.
        Y_valid is a numpy.ndarray containing the validation labels.
        layer_sizes is a list containing the number of nodes in each
                    layer of the network.
        activations is a list containing the activation functions
                    for each layer of the network.
        alpha is the learning rate.
        iterations is the number of iterations to train over.
        save_path designates where to save the model.

    Returns:
        the path where the model was saved
    """
    _, nx = X_train.shape
    _, classes = Y_train.shape

    x, y = create_placeholders(nx, classes)

    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)

        for i in range(iterations + 1):
            train_cost = session.run(loss,
                                     feed_dict={x: X_train, y: Y_train})
            train_accuracy = session.run(accuracy,
                                         feed_dict={x: X_train, y: Y_train})
            valid_cost = session.run(loss,
                                     feed_dict={x: X_valid, y: Y_valid})

            valid_accuracy = session.run(accuracy,
                                         feed_dict={x: X_valid, y: Y_valid})

            if i == iterations or i % 100 == 0:
                print("After {i} iterations:".format(i=i))
                print("\tTraining Cost: {cost}".format(cost=train_cost))
                print("\tTraining Accuracy: {accuracy}".format(
                    accuracy=train_accuracy))
                print("\tValidation Cost: {cost}".format(cost=valid_cost))
                print("\tValidation Accuracy: {accuracy}".format(
                    accuracy=valid_accuracy))

            if i < iterations:
                session.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(session, save_path)

    return save_path

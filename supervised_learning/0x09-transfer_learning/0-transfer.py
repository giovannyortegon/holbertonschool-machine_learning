#!/usr/bin/env python3
""" transfer learning """
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """ preprocess_data

    X   is a numpy.ndarray of shape (m, 32, 32, 3) containing
        the CIFAR 10 data, where m is the number of data points
    Y   is a numpy.ndarray of shape (m,) containing the CIFAR 10
        labels for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)

    return X, Y


def main():
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    X_input = K.applications.ResNet50(include_top=False,
                                      weights='imagenet',
                                      input_shape=(224, 224, 3))

    model = K.Sequentials()
    model.add(K.layers.UpSampling2D((7, 7)))
    model.add(X_input)
    model.add(K.layers.AveragePooling2D(pool_size=7))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(10, activation=('softmax')))

    checkpoint = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                             monitor='val_acc',
                                             mode='max',
                                             verbose=1,
                                             save_best_only=True)

    model.compile(optimizer=K.optimizer.RMSprop(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              batch_size=32,
              epochs=5,
              verbose=1,
              callbacks=[checkpoint])

    model.save('cifar10.h5')


if __name__ == '__main__':
    main()

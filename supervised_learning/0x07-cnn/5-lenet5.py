#!/usr/bin/env python3
""" LeNet-5 """
import tensorflow.keras as K


def lenet5(X):
    """ lenet5 - builds a modified version of the LeNet-5 architecture

    Args:
        X is a K.Input of shape (m, 28, 28, 1) containing the input images.

    Returns:
        a K.Model compiled to use Adam optimization.
    """
    c1 = K.layers.Conv2D(filters=6, kernel_size=5,
                         padding='same', activation='relu',
                         kernel_initializer='he_normal')(X)

    s2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c3 = K.layers.Conv2D(filters=16, kernel_size=5,
                         padding='valid', activation='relu',
                         kernel_initializer='he_normal')(s2)

    s4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)
    f = K.layers.Flatten()(s4)

    c5 = K.layers.Dense(units=120, activation='relu',
                        kernel_initializer='he_normal')(f)

    f6 = K.layers.Dense(units=84, activation='relu',
                        kernel_initializer='he_normal')(c5)

    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer='he_normal')(f6)

    model = K.Model(inputs=X, outputs=output)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['acccuracy'])

    return model

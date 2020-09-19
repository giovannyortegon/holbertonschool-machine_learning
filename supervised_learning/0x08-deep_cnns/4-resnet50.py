#!/usr/bin/env python3
""" resnet50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ resnet50 - builds the ResNet-50 architecture

    input data will have shape (224, 224, 3)

    Returns: the keras model
    """
    input_1 = K.layers.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(filters=64,
                             kernel_size=7,
                             padding='same',
                             strides=2,
                             kernel_initializer='he_normal')(input_1)
    batch_normalization = K.layers.BatchNormalization(axis=3)(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          padding='same',
                                          strides=(2, 2))(activation)

    proj_stg2 = projection_block(max_pooling2d, (64, 64, 256), 1)
    ident_stg2 = identity_block(proj_stg2, (64, 64, 256))
    ident_stg2a = identity_block(ident_stg2, (64, 64, 256))

    proj_stg3 = projection_block(ident_stg2a, (128, 128, 512))
    ident_stg3 = identity_block(proj_stg3, (128, 128, 512))
    ident_stg3a = identity_block(ident_stg3, (128, 128, 512))
    ident_stg3b = identity_block(ident_stg3a, (128, 128, 512))

    proj_stg4 = projection_block(ident_stg3b, (256, 256, 1024))
    ident_stg4 = identity_block(proj_stg4, (256, 256, 1024))
    ident_stg4a = identity_block(ident_stg4, (256, 256, 1024))
    ident_stg4b = identity_block(ident_stg4a, (256, 256, 1024))
    ident_stg4c = identity_block(ident_stg4b, (256, 256, 1024))
    ident_stg4d = identity_block(ident_stg4c, (256, 256, 1024))

    proj_stg5 = projection_block(ident_stg4d, (512, 512, 2048))
    ident_stg5 = identity_block(proj_stg5, (512, 512, 2048))
    ident_stg5a = identity_block(ident_stg5, (512, 512, 2048))

    average_pooling2d = K.layers.AvgPool2D(pool_size=(7, 7),
                                           padding='same')(ident_stg5a)
    dense = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer='he_normal')(average_pooling2d)
    model = K.Model(inputs=input_1, outputs=dense)

    return model

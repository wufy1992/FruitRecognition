#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import (
        Input,
        Flatten,
        Dense,
        ZeroPadding2D,
        Conv2D,
        Activation,
        MaxPooling2D,
        Dropout,
        BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU


def get_model(input_dim, category_num):
    inputs = Input(shape=(input_dim[0], input_dim[1], 3))
    x = ZeroPadding2D()(inputs)
    x = Conv2D(8, (3, 3))(x)
    x = Conv2D(16, (3, 3))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('tanh')(x)
    x = Dense(category_num)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

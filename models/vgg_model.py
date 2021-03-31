#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Reshape, Lambda, BatchNormalization, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten



def get_model(input_dim, category_num):
    """
    Build Convolution Neural Network
    args : nb_classes (int) number of classes
    returns : model (keras NN) the Neural Net model
    """
    chanDim = 1
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same",  input_shape=(input_dim[0], input_dim[1], 3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(category_num, activation='softmax'))

    return model

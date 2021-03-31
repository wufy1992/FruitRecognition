#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def get_model(input_dim, category_num):
    model = Sequential()
    model.add(Convolution2D(4, (5, 5), activation = 'tanh',input_shape=(3, input_dim[0], input_dim[1]), data_format='channels_first'))
    model.add(Convolution2D(8, (3, 3), activation = 'tanh', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Convolution2D(16, (3, 3), activation = 'tanh', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dense(category_num, init='normal'))
    model.add(Activation('softmax'))
    return model

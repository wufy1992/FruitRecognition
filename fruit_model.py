#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import os
import random
import cv2
import xmltodict
from keras.utils import np_utils
import numpy as np
from models import simple_model, ocr_model, vgg_model
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.models import load_model


class FruitModel(object):
    def __init__(self, input_dim, category_num, model_name=None):
        self.input_dim = input_dim
        self.category_num = category_num
        self.category_dict = {
            'apple': 0,
            'banana': 1,
            'orange': 2
        }
        # simple_model 500 epoch -- loss: 0.0949 - accuracy: 0.9925 - val_loss: 0.3382 - val_accuracy: 0.8866
        # ocr_model    100 epoch -- loss: 0.0076 - accuracy: 1.0000 - val_loss: 0.1220 - val_accuracy: 0.9794
        self.model = ocr_model.get_model(input_dim, category_num)

    def train_and_save_model(self, train_data_dir, test_dir, model_path):
        data, label = self.load_data(train_data_dir)
        test_data, test_label = self.load_data(test_dir)
        label = np_utils.to_categorical(label, self.category_num)
        test_label = np_utils.to_categorical(test_label, self.category_num)
        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        self.model.fit(data, label, batch_size=32, epochs=300, verbose=1,  validation_data=(test_data, test_label))
        self.model.save(model_path)

    def load_data(self, data_dir):
        data_list = list()
        label_list = list()
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.jpg'):
                fruit_name = file_name.split('_')[0]
                if fruit_name in self.category_dict:
                    file_path = os.path.join(data_dir, file_name)
                    img = cv2.imread(file_path)
                    xml_path = os.path.join(data_dir, file_name.replace('jpg', 'xml'))
                    if os.path.exists(xml_path):
                        xml_str = open(xml_path, "r").read()
                        xml_dict = xmltodict.parse(xml_str)
                        objects = xml_dict['annotation']['object']
                        if not isinstance(objects, list):
                            objects = [objects]
                        for object_node in objects:
                            fruit_name = object_node['name']
                            bndbox = object_node['bndbox']
                            fruit_image = img[int(bndbox['ymin']):int(bndbox['ymax']), int(bndbox['xmin']):int(bndbox['xmax'])]
                            fruit_image = cv2.resize(fruit_image, (self.input_dim[0], self.input_dim[1]))
                            data_list.append(fruit_image)
                            label_list.append(self.category_dict[fruit_name])
        data_np = np.array(data_list)
        label_np = np.array(label_list)
        index = [i for i in range(len(data_np))]
        random.shuffle(index)
        data_np = data_np[index]
        label_np = label_np[index]
        return data_np, label_np


if __name__ == "__main__":
    model_path = 'models/ocr_model.h5'
    test_model = FruitModel([128, 128], 4)
    if os.path.exists(model_path):
        test_model.model = load_model(model_path)
    test_model.train_and_save_model('train_data', 'test_data', model_path)


# -*- coding: utf-8 -*-
# @Date               : 2021-01-29 16:56:36
# @Author             : Haoyu Kong, WZMIAOMIAO
# @Python version     : 3.6.12
# @Tensorflow version : 2.0.0
# @cudatoolkit version: 10.0.130
# @cudnn version      : 7.6.5

# When performing model training, please make sure that this script and
# the training script are in the same working directory.

import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential, regularizers

# Models structure. "64", "128", "256" and "512" are the number of channel. "M" stands for "maxpooling"
cfgs = {
    'vgg8': [64,'M',128,'M',256,'M',512,'M',512,'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# Based on the models' structure, creating the feature extraction layers and encapsulated them into a Sequential
def features(cfg):
    feature_layers = []
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'))
        else:
            conv2d = layers.Conv2D(v, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name="feature")


# Establish the complete VGGNet model. Add the fully connected layers and the dropout layers after the Sequential of feature extraction
# im_height, im_width: The height and width of the input images
# class_num: Number of classes in the classification task
def VGG(feature, im_height=224, im_width=224, class_num=2):
    input_image = layers.Input(shape=(im_height, im_width, 1), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(4096, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(4096, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001))(x)
    output = layers.Dense(class_num, activation=None)(x)
    model = models.Model(inputs=input_image, outputs=output)
    return model


#According to the input model configuration name, output a complete VGGNet model.
def vgg(model_name="vgg16", im_height=224, im_width=224, class_num=2):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(features(cfg), im_height=im_height, im_width=im_width, class_num=class_num)
    return model


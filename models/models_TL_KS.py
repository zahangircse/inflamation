# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:51:46 2020

@author: md zahangir alom
"""
"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
import os
import warnings
#import tensorflow as tf
#import keras
import pdb
   
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np

#from tensorflow.keras.applications import EfficientNetB0

#from keras_efficientnets.efficientnet import EfficientNetB0

#from keras_efficientnets import EfficientNetB5

#preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

backend = None
layers = None
models = None
keras_utils = None


#def build_EN_model(input_shape,n_out_classes):
#    
#    input_tensor = keras.Input(shape=input_shape)
#    base_model = EfficientNetB0(input_shape, classes=n_out_classes, include_top=False, pooling = 'avg', weights=None)
#    #model = efficientnet.EfficientNetB5(input_shape, classes=n_out_classes, include_top=False, pooling = 'avg', weights=None)
#    
#    print("[INFO] summary for base model...")
#    print(base_model.summary())
#    
#    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
#    x = keras.layers.Dropout(0.5)(x)
#    x = keras.layers.Dense(1024, activation='relu')(x)
#    x = keras.layers.Dropout(0.5)(x)
##     x = Dense(1024, activation='relu')(x)
##     x = Dropout(0.3)(x)
#    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
#    model = keras.Model(input_tensor, final_output)
#    
#    return base_model, model



def VGG16_TL_model(input_shape,n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.VGG16(include_top=False,
                   weights="imagenet",
                   input_tensor=input_tensor)

    print("[INFO] summary for base model...")
    print(base_model.summary())
    
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.3)(x)
    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model
    
def VGG19_TL_model(input_shape,n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.VGG19(include_top=False,
                   weights="imagenet",
                   input_tensor=input_tensor)

    print("[INFO] summary for base model...")
    print(base_model.summary())
    
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.3)(x)
    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model

# def build_ResNet34_TL_model(input_shape, n_out_classes):
    
#     input_tensor = keras.Input(shape=input_shape)
#     base_model = keras.applications.resnet34.ResNet32(include_top=False,
#                    weights="imagenet",
#                    input_tensor=input_tensor)

#     print("[INFO] summary for base model...")
#     print(base_model.summary())
    
#     x = keras.layers.GlobalAveragePooling2D()(base_model.output)
#     x = keras.layers.Dropout(0.5)(x)
#     x = keras.layers.Dense(1024, activation='relu')(x)
#     x = keras.layers.Dropout(0.5)(x)
# #     x = Dense(1024, activation='relu')(x)
# #     x = Dropout(0.3)(x)
#     final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
#     model = keras.Model(input_tensor, final_output)
    
#     return base_model, model


def build_ResNet50_TL_model(input_shape, n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                   weights="imagenet",
                   input_tensor=input_tensor)

    print("[INFO] summary for base model...")
    print(base_model.summary())
    
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.3)(x)
    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model

def build_ResNet50_model(input_shape,n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                   weights=None,input_tensor = input_tensor)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model
    

def build_InceptionV3_TL_model(input_shape, n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.InceptionV3(include_top=False,
                   weights="imagenet",
                   input_tensor=input_tensor)

    print("[INFO] summary for base model...")
    print(base_model.summary())
    
    #pdb.set_trace()
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.3)(x)
    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model

def build_InceptionV3_model(input_shape,n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    
    base_model = keras.applications.InceptionV3(include_top=False,
                   weights=None,input_tensor = input_tensor)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model


def build_InceptionV3_TL_keras_model(input_shape, n_out_classes):
    
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.InceptionV3(include_top=False,
                   weights="imagenet",
                   input_tensor=input_tensor)

    print("[INFO] summary for base model...")
    print(base_model.summary())
    
    #pdb.set_trace()
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.3)(x)
    final_output = keras.layers.Dense(n_out_classes, activation='softmax', name='final_output')(x)
    model = keras.Model(input_tensor, final_output)
    
    return base_model, model




def small_model(input_shape, num_classes):

     
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
     
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
     
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
     
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
     
    #model.summary()
    
    return model 


def small_model_128(input_shape, num_classes):

     
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
     
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
     
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
     
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
     
    #model.summary()
    
    return model 
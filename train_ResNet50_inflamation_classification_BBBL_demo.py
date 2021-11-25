from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
#import tensorflow.contrib.slim as slim
#from models import models_TL as models
from models import models_TL_KS as path_models
import numpy as np
import time, datetime
import argparse
import random
import os,sys,json
import subprocess
from os.path import join as join_path
import glob
import cv2
import keras
#import tensorflow as tf
#from tensorflow import keras
from pathlib import Path

from keras.optimizers import SGD
#from keras.utils import multi_gpu_model
import matplotlib
import pdb
matplotlib.use('Agg')

from utils import utils as utils
from utils import helpers as helpers

from models import models_TL_KS as TL_models
from keras.callbacks import ModelCheckpoint
#from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

abspath = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def flipRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="project_inflamation_ResNet50_BBBL", help='Name of your project')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--num_GPUs', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset_path_train', type=str, default="~training_dataset_path~/training/", help='Dataset you are using.')
parser.add_argument('--dataset_path_val', type=str, default="~val_dataset_path/val/", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')

args = parser.parse_args()
# make necessary directory for storing log files...
if not os.path.isdir("%s/%s"%("experimental_logs",args.project_name)):
    os.makedirs("%s/%s"%("experimental_logs",args.project_name))

project_path = join_path('experimental_logs/', args.project_name+'/')

if not os.path.isdir("%s/%s"%(project_path,"training")):
    os.makedirs("%s/%s"%(project_path,"training"))
    os.makedirs("%s/%s"%(project_path,"testing"))
    os.makedirs("%s/%s"%(project_path,"weights"))
    #os.makedirs("%s/%s"%(project_path,"train_test_set"))
    os.makedirs("%s/%s"%(project_path,"train_model"))
    os.makedirs("%s/%s"%(project_path,"extracted_features"))

# create all necessary path for saving log files 
training_log_saving_path = join_path(project_path,'training/')
testing_log_saving_path = join_path(project_path,'testing/')
weight_saving_path = join_path(project_path,'weights/')
traned_model_saving_path = join_path(project_path,'train_model/')
features_saving_path = join_path(project_path,'extracted_features/')
# Load the data, normalization, and spliting to training, validation, and testing sets...
print("Data preparation and Loading ... ...")
image_size = (args.crop_height, args.crop_width)

# Load the data
print("Loading the data for training ...")

# augmentation of images settings
image_data_gen_configs = {
        'rescale': 1 / 255,
        'rotation_range': 360,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'preprocessing_function': flipRGB
    }

classes = os.listdir(args.dataset_path_train)

num_train_images = len(list(Path(args.dataset_path_train).glob('**/*'))) - len(classes)
num_test_images = len(list(Path(args.dataset_path_val).glob('**/*'))) - len(classes)

train_gen = ImageDataGenerator(**image_data_gen_configs)
val_gen = ImageDataGenerator(rescale=1 / 255, preprocessing_function=flipRGB)
      
#x_data,y_data,y_names,tags = data_utils.read_traning_data_4classificaiton_wNames(args.dataset_path_train,args.crop_height, args.crop_width)
num_classes = len(classes)


# training images
train_generator = train_gen.flow_from_directory(
        args.dataset_path_train,
        target_size=(args.crop_height, args.crop_width),
        classes=classes,
        batch_size=args.batch_size,
        class_mode='categorical')

# validation images
validation_generator = val_gen.flow_from_directory(
        args.dataset_path_val,
        target_size=(args.crop_height, args.crop_width),
        classes=classes,
        batch_size=args.batch_size,
        class_mode='categorical')

num_channels = 3    
net_input = (args.crop_height, args.crop_width,num_channels)

num_gpus_to_use = args.num_GPUs 
print("\n ***** Training details *****")
print("Number GPUs in use -->", args.num_GPUs )
print("Dataset -->", args.dataset_path_train)
#print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)
print("\n ** Training model ...")

# load model here..
input_shape = net_input
inlucde_top = False
weights = 'imagenet'

include_top=True 
input_shape=net_input
pooling='avg'
classes=num_classes
base_model, model = path_models.build_ResNet50_model(input_shape, classes)
model.summary()

parallel_model = model
checkpoint = ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_loss:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
#parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=["accuracy"])

history = parallel_model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_images // args.batch_size,
        epochs=args.num_epochs,
        verbose=2,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=num_test_images // args.batch_size#,
        #class_weight=train_helper.get_class_weights(train_dir)  # weighted train to better handle any class imbalance
    )
    
final_model_sv_path = os.path.join(traned_model_saving_path,'final_model.h5')
parallel_model.save(final_model_sv_path)

# saving the log values ....
training_log = {}
training_log["Model_loss"] = history.history['loss']
training_log["accuracy"] = history.history['accuracy']
training_log["val_loss"] = history.history['val_loss']
training_log["val_accuracy"] = history.history['val_accuracy']
#training_log["TTT"] = total_time
json_file = os.path.join(training_log_saving_path,'training_log_class91_final.json')
with open(json_file, 'w') as file_path:
    json.dump(training_log, file_path, indent=4, sort_keys=True)


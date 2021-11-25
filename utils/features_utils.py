#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:30:37 2020

@author: malom
"""

import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import cv2
from os.path import join as join_path
import pdb
from collections import defaultdict
from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage



def dl_feature_saving(feature_saving_path,feature_type,ac_labels,dl_features):
    
    #np.savez('mat.npz', name1=arr1, name2=arr2)
    #data = np.load('mat.npz')
    #print data['name1']
    #print data['name2']

    name = feature_type+'.npy'
    saving_path = join_path(feature_saving_path,name)
    np.save(saving_path,dl_features)
    
    label_name = feature_type+'_ac_labels.npy'
    label_saving_path = join_path(feature_saving_path,label_name)
    np.save(label_saving_path,ac_labels)

    
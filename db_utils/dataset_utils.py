#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:57:07 2018
@author: zahangir
"""
import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import cv2
from os.path import join as join_path
import pdb
from collections import defaultdict
#from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage
import tensorflow as tf
import json
import scipy
from json import JSONEncoder
from sklearn import preprocessing


kernel = np.ones((6,6), np.uint8) 

allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif']

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def color_perturbation(image):
      
      image = tf.image.random_brightness(image, max_delta=64./ 255.)
      image = tf.image.random_saturation(image, lower=0, upper=0.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, lower=0, upper=0.75)      
      
      return image

def flipRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def labels_encoder(acc_labels):
    encoded_obj = preprocessing.LabelEncoder()
    encoded_obj.fit(acc_labels)
    labels_encoded = encoded_obj.transform(acc_labels)
    return encoded_obj, labels_encoded
    
def labels_decoder(decoder_object,model_pred):
    
    decoded_pred = decoder_object.inverse_transform(model_pred)
    
    return decoded_pred

def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess_input_norm_sub_mean(x0):
    x = x0 / 255.
    x -= 0.5
    return x

def preprocess_input_norm(x0):
    x = x0 / 255.
    return x
    

def samples_normalization (x_data, y_data):
    x_data = x_data.astype('float32')
    mean = np.mean(x_data)  # mean for data centering
    std = np.std(x_data)  # std for data normalization
    x_data -= mean
    x_data /= std
    
    y_data = y_data.astype('float32')
    y_data /= 255.  # scale masks to [0, 1]
    return x_data,y_data,mean,std


def shuffle_input_samples(data_x,data_y,data_y_encoded):
    ## data_x and data_y are numpy array    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_y_encoded = np.array(data_y_encoded)

    perm = np.random.permutation(len(data_y))
    X = data_x[perm]
    y = data_y[perm]
    y_encoded = data_y_encoded[perm]
    
    return X,y,y_encoded
    
def shuffling_samples(data_x,data_y,data_y_encoded):    
    ## data_x and data_y are numpy array    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_y_encoded = np.array(data_y_encoded)

    perm = np.random.permutation(len(data_y))
    X = data_x[perm]
    y = data_y[perm]
    y_encoded = data_y_encoded[perm]
    
    return X,y,y_encoded
    
    
def unique_finder(list1):       
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set))    
    unique_labels = []
    for x in unique_list:        
        unique_labels.append(x)
        print(x)
       
    unique_labels = np.array(unique_labels)
    return unique_labels
        
def classe_logs(labels):
    
    classes_name = unique_finder(labels)
    num_classes = len(np.unique(labels))
    return classes_name, num_classes

def save_database_logs(y_ac_data,database_log_saving_path):
    
    number_sampels = len(y_ac_data)
    classes = np.unique(y_ac_data)
    num_classes = len(classes)
    print("Number of classes is : "+str(classes))   
    
    label_list = y_ac_data.tolist()
    
    sample_count_per_class = []
    
    for i, idv_label in enumerate(classes):     
        ns_per_class = label_list.count(idv_label)
        sample_count_per_class.append(str(idv_label)+' = '+str(ns_per_class))
    

    database_log = {}
    database_log["Total samples"] = number_sampels
    database_log["Total_classes"] = num_classes
    database_log["Labels name"] = classes
    database_log["Sample logs"] = sample_count_per_class
     
    #database_log_list = database_log.tolist()
    # make experimental log saving path...
    #pdb.set_trace()
    json_file = os.path.join(database_log_saving_path,'database_logs.json')
    with open(json_file, 'w') as file_path:
        json.dump(database_log, file_path, cls=NumpyArrayEncoder)  
        
    classes_name = classes
        
    return classes_name,num_classes,number_sampels

def split_data_train_val(x_data,y_data,y_ac_data):

    sample_count = len(x_data)   
    train_size = int(sample_count * 4.8 // 5)    
    
    #ac_x_train = ac_x_data[:train_size]
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    y_ac_train = y_ac_data[:train_size]
    
    #ac_x_val = ac_x_data[train_size:]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    y_ac_val = y_ac_data[train_size:]

    
    return x_train,y_train,y_ac_train, x_val,y_val, y_ac_val


def STN_and_return(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Io = 240 # Transmitted light intensity, Normalizing factor for image intensities
    alpha = 1  #As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
    beta = 0.15 #As recommended in the paper. OD threshold for transparent pixels (default: 0.15)
    ######## Step 1: Convert RGB to OD ###################
    ## reference H&E OD matrix.
    #Can be updated if you know the best values for your image. 
    #Otherwise use the following default values. 
    #Read the above referenced papers on this topic. 
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])
    # extract the height, width and num of channels of image
    h, w, c = img.shape
    # reshape image to multiple rows and 3 columns.
    #Num of rows depends on the image size (wxh)
    img = img.reshape((-1,3))
    # calculate optical density
    # OD = −log10(I)  
    #OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    #Adding 0.004 just to avoid log of zero. 
    OD = -np.log10((img.astype(np.float)+1)/Io) #Use this for opencv imread

    #Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)
    ############ Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta
    #Check by printing ODhat.min()
    ############# Step 3: Calculate SVD on the OD tuples ######################
    #Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))    
    #eigvals, eigvecs = scipy.linalg.eigvalsh(np.cov(ODhat.T)) 
    ######## Step 4: Create plane from the SVD directions with two largest values ######
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3]) #Dot product
    ############### Step 5: Project data onto the plane, and normalize to unit length ###########
    ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
    #find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    ###### Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix 
    
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    #plt.imsave(path_outputs, Inorm)
    return Inorm


def read_classification_samples_colon_cancer(data_path,image_size):
    
    data_path = join_path(data_path,'/*')
    SIZE = image_size
    train_images = []
    train_labels = [] 
    for directory_path in glob.glob(data_path):
        label = directory_path.split("/")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
            print(img_path)
            crop_img = np.ndarray((SIZE, SIZE,3), dtype=np.uint8)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            h,w,c = img.shape
            crop_img[:,:,:] = img[36:h-36,36:w-36,:]
            #img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label)
            
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    return train_images,train_labels

def read_classification_samples(data_path,image_size):
    
    SIZE = image_size
    train_images = []
    train_labels = [] 
    for directory_path in glob.glob(data_path):
        label = directory_path.split("/")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label)
            
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    return train_images,train_labels

def applyImageAugmentationAndRetrieveGenerator():


    # We create two instances with the same arguments
    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2
                         )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory('dataset/train_images',
                                                        target_size=(360,480),    
                                                        class_mode=None,
                                                        seed=seed,
                                                        batch_size = 32)
    
    mask_generator = mask_datagen.flow_from_directory('dataset/train_masks',
                                                      target_size=(360,480),  
                                                      class_mode=None,
                                                      seed=seed,
                                                      batch_size = 32)
    

    train_generator = zip(image_generator, mask_generator)
    
    return train_generator

def extract_image_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):
        
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)

    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
            
            patch_img_flip_lr = np.fliplr(patch_img)         
            patch_mask_flip_lr = np.fliplr(patch_mask)
            
            patch_img_flip_up = np.flipud(patch_img)
            patch_mask_flip_up = np.flipud(patch_mask)
            
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.jpg'           
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)
            
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
            
            f_img_name_lr =str(img_name)+'_'+str(pn)+'_lr.jpg'
            f_mask_name_lr =str(img_name)+'_'+str(pn)+'_lr_mask'+'.jpg'           
            final_des_img_lr = os.path.join(imd_saving_dir,f_img_name_lr)
            final_des_mask_lr = os.path.join(imd_saving_dir,f_mask_name_lr)
            
            
            cv2.imwrite(final_des_img_lr,patch_img_flip_lr)
            cv2.imwrite(final_des_mask_lr,patch_mask_flip_lr)
            
            
            f_img_name_up =str(img_name)+'_'+str(pn)+'_up.jpg'
            f_mask_name_up =str(img_name)+'_'+str(pn)+'_up_mask'+'.jpg'           
            final_des_img_up = os.path.join(imd_saving_dir,f_img_name_up)
            final_des_mask_up = os.path.join(imd_saving_dir,f_mask_name_up)
            
              
            cv2.imwrite(final_des_img_up,patch_img_flip_up)
            cv2.imwrite(final_des_mask_up,patch_mask_flip_up)
            
            #mx_val = patch_mask.max()
            #mn_val = patch_mask.min()
            #print ('max_val : '+str(mx_val))
            #print ('min_val : '+str(mn_val))          

            #if mx_val > 10:

            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))

    return pn

def read_single_pixel_anno_data(image_dir,img_h,img_w):

    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.jpg']
    
    total = int(np.round(len(all_images)/2))

    ac_imgs = np.ndarray((total, img_h,img_w,3), dtype=np.uint8)
    imgs = np.ndarray((total, img_h,img_w), dtype=np.uint8)
    imgs_mask = np.ndarray((total,img_h,img_w), dtype=np.uint8)
    k = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for i, image_name in enumerate(all_images):
         if 'mask' in image_name:
             continue
         image_mask_name = image_name.split('.')[0] + '_mask.jpg'
          # patient_num = image_name.split('_')[0]
         img = cv2.imread(os.path.join(image_dir, image_name), cv2.IMREAD_GRAYSCALE)
         ac_img = cv2.imread(os.path.join(image_dir, image_name))
         img_mask = cv2.imread(os.path.join(image_dir, image_mask_name), cv2.IMREAD_GRAYSCALE)
         img_mask = 255.0*(img_mask[:,:]> 0)
         img_mask = cv2.dilate(img_mask,kernel,iterations = 1)
         img_mask = ndimage.gaussian_filter(img_mask, sigma=(1,1),order = 0)     
         img_mask = 255.0*(img_mask[:,:]> 0)
        
         
         ac_imgs[k] = ac_img 
         imgs[k] = img
         imgs_mask[k] = img_mask

         k += 1
         print ('Done',i)
     
    """
    perm = np.random.permutation(len(imgs_mask))
    imgs = imgs[perm]
    imgs_mask = imgs_mask[perm]
    ac_imgs = ac_imgs[perm]
    """
    return ac_imgs, imgs, imgs_mask

def create_dataset_patches_driver(image_dir,saving_dir,patch_h,patch_w):
        
    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.jpg']
    

    pdb.set_trace()
     
    Total_patches = 0

    for i, name in enumerate(all_images):
        
        if 'morh_banary' in name:
            continue
          
        im = cv2.imread(image_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    
        #im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
        acc_name = name.split('.')[0]
        mask_name = acc_name +'_morh_banary.jpg'       
        mask_im = cv2.imread(image_dir + mask_name, cv2.IMREAD_UNCHANGED) #.astype('float32')/255.
        #mask_im = cv2.resize(mask_im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        #y_data[i] = mask_im
        
        mask_im = 255*(mask_im[:,:]> 20)
        img_rz = im
        img_mask_rz = mask_im
        
       
        
        num_patches = extract_image_patches (img_rz, img_mask_rz, patch_h, patch_w, acc_name, saving_dir)
        
        print ('Processing for: ' +str(i))
        Total_patches = Total_patches + num_patches
    
    return 0


def read_testing_images(data_path,image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    #images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))
    images = glob.glob(train_data_path + "/*.jpg")
    total = np.round(len(images)) 

    acc_imgs = np.ndarray((total, image_h, image_w,3), dtype=np.uint8)
    gray_mgs = np.zeros((total, image_h, image_w), dtype=np.uint8)
    #imgs_mask = np.zeros((total, image_h, image_w), dtype=np.uint8)
    
    i = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:
        '''
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        '''
        #image_mask_name = image_name.split('/')[-1]      
        #img_first = image_mask_name.split('.')[0]
        #img_second = img_first.split('_mask')[0]      
        #image_name =img_second+'.jpg'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        gray_img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        #img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        acc_imgs[i] = acc_img
        gray_mgs[i] = gray_img
        #imgs_mask[i] = img_mask

        i += 1
        print ('Done',i)
    
    return acc_imgs, gray_mgs

def read_images_and_masks(data_path, image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    images = glob.glob(train_data_path + "/*mask.png")
    total = np.round(len(images)) 

    acc_imgs = np.ndarray((total, image_h, image_w,3), dtype=np.uint8)
    imgs = np.zeros((total, image_h, image_w), dtype=np.uint8)
    imgs_mask = np.zeros((total, image_h, image_w), dtype=np.uint8)
    
    i = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:

        image_mask_name = image_name.split('/')[-1]      
        img_first = image_mask_name.split('.')[0]
        img_second = img_first.split('_mask')[0]      
        image_name =img_second+'.png'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        acc_imgs[i] = acc_img
        imgs[i] = img
        imgs_mask[i] = img_mask

        i += 1
        print ('Done',i)
    
    return acc_imgs,imgs,imgs_mask


def read_traning_data_4classificaiton(base_dir, h,w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            #print(img_name_1)
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='jpg' or 'png':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                X.append(img)
                y.append(class_index)
                
                useful_image_count += 1
        

    X = np.array(X).astype(np.float32)
    #X = X.transpose((0, 3, 1, 2))
    X=X.transpose((0,1,2,3))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags

def read_traning_data_4classificaiton_wNames(base_dir, img_h,img_w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())
    processed_image_count = 0
    useful_image_count = 0
    X = []
    y = []
    y_name = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        print('Data is processing for the class of ',class_name)
        
        for filename in filenames:
            processed_image_count += 1
            print(filename)
            img_name_1 = filename.split('/')[-1]
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            print('Reading the following image:'+img_name)            
            if img_extension == 'png': 
                img = cv2.imread(filename)
                #img= np.array(img)  
                img = cv2.resize(img, (img_h,img_w), interpolation = cv2.INTER_AREA)
                img = preprocess_input_norm(img)
                X.append(img)
                y.append(class_index) 
                y_name.append(img_name)                
                useful_image_count += 1
        
    #pdb.set_trace()
    X = np.array(X).astype(np.float32)
    # X=X.transpose((0,1,2,3))
    # y = np.array(y) 
    # #pdb.set_trace()
    # perm = np.random.permutation(len(y))
    # X = X[perm]
    # y = y[perm]
    # y_name = y_name[perm]

    #print("classes:")
    #for class_index, class_name in enumerate(tags):
    #    print(class_name, sum(y == class_index))
    
    #print("\n")

    return X, y, y_name, tags

def read_traning_data_4classificaiton_wNames_infm_aug(base_dir, img_h,img_w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())
    processed_image_count = 0
    useful_image_count = 0
    X = []
    y = []
    y_name = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        print('Data is processing for the class of ',class_name)
        
        for filename in filenames:
            processed_image_count += 1
            print(filename)
            img_name_1 = filename.split('/')[-1]
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            print('Reading the following image:'+img_name)            
            if img_extension == 'png': 
                img = cv2.imread(filename)
                #img= np.array(img)  
                img = cv2.resize(img, (img_h,img_w), interpolation = cv2.INTER_AREA)
                #img = preprocess_input_norm(img)
                #X.append(img)
                #y.append(class_index) 
                useful_image_count += 1
                
                if class_index ==1 :
                    print('Reading inflamation and performing Aug.')
                    X.append(img)
                    y.append(class_index)
                    y_name.append(img_name)                

#                    flip_lr = np.fliplr(img)
#                    X.append(flip_lr)
#                    y.append(class_index)
#                    y_name.append(img_name.split('.')[0]+'_lr')                
#
#                    flip_ud = np.flipud(img)
#                    X.append(flip_ud)
#                    y.append(class_index)
#                    y_name.append(img_name.split('.')[0]+'_ud')                

                    useful_image_count += 1
                else:
                    X.append(img)
                    y.append(class_index)
                    y_name.append(img_name)                
                    useful_image_count += 1
                  
        
    #pdb.set_trace()
    X = np.array(X).astype(np.float32)
    # X=X.transpose((0,1,2,3))
    # y = np.array(y) 
    # #pdb.set_trace()
    # perm = np.random.permutation(len(y))
    # X = X[perm]
    # y = y[perm]
    # y_name = y_name[perm]

    #print("classes:")
    #for class_index, class_name in enumerate(tags):
    #    print(class_name, sum(y == class_index))
    
    #print("\n")

    return X, y, y_name, tags


def read_traning_data_4classificaiton_wNames_SN(base_dir, img_h,img_w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())
    processed_image_count = 0
    useful_image_count = 0
    X = []
    y = []
    y_name = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        print('Data is processing for the class of ',class_name)
        
        for filename in filenames:
            processed_image_count += 1
            print(filename)
            img_name_1 = filename.split('/')[-1]
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            print('Reading the following image:'+img_name)            
            if img_extension == 'png': 
                img = cv2.imread(filename)
                #img= np.array(img)  
                img = cv2.resize(img, (img_h,img_w), interpolation = cv2.INTER_AREA)
                img = STN_and_return(img)
                X.append(img)
                y.append(class_index) 
                y_name.append(img_name)                
                useful_image_count += 1
        
    #pdb.set_trace()
    X = np.array(X).astype(np.float32)
    # X=X.transpose((0,1,2,3))
    # y = np.array(y) 
    # #pdb.set_trace()
    # perm = np.random.permutation(len(y))
    # X = X[perm]
    # y = y[perm]
    # y_name = y_name[perm]

    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, y_name, tags

def read_traning_data_4classificaiton_withAug(base_dir, h, w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            #print(img_name_1)
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='png' or 'jpg':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                
                if class_index ==0:
                    X.append(img)
                    y.append(class_index)
                    useful_image_count += 1
                else:
                    X.append(img)
                    y.append(class_index)
                
                    flip_lr = np.fliplr(img)
                    X.append(flip_lr)
                    y.append(class_index)
                    
                    flip_ud = np.flipud(img)
                    X.append(flip_ud)
                    y.append(class_index)
                    
                    useful_image_count += 3
        

    X = np.array(X).astype(np.float32)
    #X = X.transpose((0, 3, 1, 2))
    X=X.transpose((0,1,2,3))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags


def extract_random_patches_masks_from_images(full_imgs,mask_img,central_xy, N_patches, patch_h,patch_w, img_name, imd_saving_dir):
       
    central_xy.astype(int)    
    height, width, chan = full_imgs.shape   
    start_x = patch_h/2
    start_y = patch_w/2
    end_x = height-start_x
    end_y = width - start_y
    
    k=0
    pn = 0
    while k <N_patches:
        x_center = int(central_xy[k,0])
        y_center =  int(central_xy[k,1])
        
        if (x_center > start_x and y_center > start_y and x_center < end_x and y_center < end_y) :
            img_patch = full_imgs[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2),:]
            mask_patch = mask_img[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]

            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask.jpg'

            final_img_des = os.path.join(imd_saving_dir,f_img_name)
            final_mask_des = os.path.join(imd_saving_dir,f_mask_name)
            

            mx_val = final_mask_des.max()
            mn_val = final_mask_des.min()
            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))
            #if mx_val > 10:
            #    cv2.imwrite(final_img_des,img_patch)
            #    cv2.imwrite(final_mask_des,mask_patch)             
            cv2.imwrite(final_img_des,img_patch)
            cv2.imwrite(final_mask_des,mask_patch)

            pn +=1
   
        k +=1  

    print ('Processing for: ' +str(k))
    
    return pn

            
def extract_random_patches_from_images(full_imgs,central_xy, N_patches, patch_h,patch_w, img_name, imd_saving_dir):
       
    central_xy.astype(int)    

    k=0   # for the loop...
    pn = 0
    while k <N_patches:
        x_center = int(central_xy[k,0])
        y_center =  int(central_xy[k,1])      
        img_patch = full_imgs[y_center:y_center+patch_h,x_center:x_center+patch_w,:]
        f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
        final_img_des = os.path.join(imd_saving_dir,f_img_name)          
        cv2.imwrite(final_img_des,img_patch)
        pn +=1
        k +=1   # for the loop 
        print ('Saving the patch of: ',img_name+'_'+str(pn))
    
    return pn

def create_dataset_random_patches_driver(data_path,patch_h,patch_w,number_samples_per_images, data_saving_dir):
    
    #path, subdirs, files = os.walk(data_path)
    
    for path, subdirs, files in os.walk(data_path):        
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print(name)   
            
            if not os.path.isdir("%s/%s"%(data_saving_dir,name)):
                os.makedirs("%s/%s"%(data_saving_dir,name))
            data_saving_dir_final = join_path(data_saving_dir,name+'/')
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.tif']    
            
            for filename in images:                  
                image_path = os.path.join(sub_dir,filename)           
                img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
                #img_name = filename.split('/')[-1]
                img_name = filename.split('.')[0]+'_'+filename.split('.')[1]

                # Create random point pair with respect to the size of the images...
                height, width, chan = img.shape                  
                start_x = patch_h
                start_y = patch_w
                end_x = height-start_x
                end_y = width - start_y              
                radius = min((end_x, end_y))   
        
                central_xy = np.random.random((number_samples_per_images, 2))*radius                     
                central_xy=central_xy.astype(int)
                print(img_name)
                Num_samples = len(central_xy)
                
                                
                #pdb.set_trace()
                
                if Num_samples > 0:
                       num_patches = extract_random_patches_from_images(img, central_xy, Num_samples, patch_h, patch_w, img_name, data_saving_dir_final)

    return 0

                        
def create_dataset_random_patches_masks_driver(base_dir,patch_h,patch_w, number_samples_per_images, data_saving_dir):
    
    train_data_path = os.path.join(base_dir)
    images = filter((lambda image: '_anno' not in image), os.listdir(train_data_path))
    total = np.round(len(images)) 

    pdb.set_trace()
    
    for filename in images:

        #print(filename)     
        img = cv2.imread(os.path.join(base_dir,filename))            
        #img_path_first= os.path.dirname (filename)
            #img_path_first =filename.split('.')[0]
        img_name = filename.split('/')[-1]
        img_name = img_name.split('.')[0]
        mask_name = img_name+'_anno.bmp'
            #mask_name = '/'+mask_name
        mask_path = os.path.join(base_dir,mask_name)           
        mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        
        # show heatmaps image..
        #plt.imshow(mask_img, cmap='hot', interpolation='nearest')
        #plt.show() 
        #mask_img = 255*(mask_img[:,:]>0)
                    
        height, width, chan = img.shape           
        radius = min((height, width))           
        central_xy = np.random.random((number_samples_per_images, 2))*radius 
            
        central_xy=central_xy.astype(int)
           # print(class_index)
        print(img_name)
                        
        img_saving_dir_b = os.path.join(data_saving_dir,'images_and_masks_benign_malignant/benign/')  
            #img_saving_dir_m = os.path.join(data_dir,'images_and_masks_benign_malignant/malignant/') 

            #Extract random patches from image for each 
        if len(central_xy) > 0:
               num_patches = extract_random_patches_masks_from_images (img, mask_img, central_xy, len(central_xy), patch_h, patch_w, img_name, img_saving_dir_b)

    return 0



# data_path = '/home/malom/stjude_projects/digital_path/colon_cancer_detection/database/train_val/*'
# SIZE = image_size[0]

# train_images = []
# train_labels = [] 

# h = 150
# w = 150

# e_pd = (h - args.crop_height)
# e_p = int(e_pd/2)
# print('Cropping point start :', e_p)
# #pdb.set_trace()

# for directory_path in glob.glob(data_path):
#     label = directory_path.split("/")[-1]
#     print(label)
#     for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
#         print(img_path)
#         crop_img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
#         h,w,c = img.shape
#         crop_img[:,:,:] = img[e_p:h-e_p,e_p:w-e_p,:]
#         #img = cv2.resize(img, (SIZE, SIZE))
#         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
#         train_images.append(crop_img)
#         train_labels.append(label)
        
# data_x = np.array(train_images)
# data_y = np.array(train_labels)

# classes_name, num_classes = data_utils.classe_logs(data_y)
#x_data,y_data = data_utils.samples_normalization (x_data, y_data)
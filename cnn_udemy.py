# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:20:40 2019

@author: HP
"""

import tensorflow as tf 
import keras 

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier=Sequential()
#step1:convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#step2:pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

####adding 2nd convolution
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3:flattening
classifier.add(Flatten())

#step4:full connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compliing the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])






from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

trainning_set=train_datagen.flow_from_directory('C:\\Users\\HP\\Desktop\\u_datasets\\Convolutional_Neural_Networks\\dataset\\training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set=test_datagen.flow_from_directory('C:\\Users\\HP\\Desktop\\u_datasets\\Convolutional_Neural_Networks\\dataset\\test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')

classifier.fit_generator(trainning_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)


import numpy as np 
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\HP\\Desktop\\u_datasets\\Convolutional_Neural_Networks\\dataset\\single_prediction\\cat_or_dog_1.jpg', target_size = (64, 64)) 
# This is test img 
# first arg is the path 
# img is 64x64 dims this is what v hv used in training so wee need to use exactly the same dims 
# here also 
 
test_image 
 
 

test_image = image.img_to_array(test_image) 
# Also in our first layer below it is a 3D array 
# Step 1 - Convolution 
# classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) 
# this will convert from a 3D img to 3D array 
test_image # shld gv us (64,64,3) 
 
 
 
 
test_image = np.expand_dims(test_image, axis = 0) 
# axis specifies the position of indx of the dimnsn v r addng 
# v need to add the dim in the first position 
test_image # now it shld show (1,64,64,3) 
 
 
 
result = classifier.predict(test_image) 
# v r trying to predict 
result # gv us 1 
 
 
print(trainning_set.class_indices) 
 
if result[0][0] == 1: 
    prediction = 'dog' 
else: 
    prediction = 'cat' 

 
#import numpy as np 
#from keras.preprocessing import image 
    
test_image = image.load_img('C:\\Users\\HP\\Desktop\\u_datasets\\Convolutional_Neural_Networks\\dataset\\single_prediction\\cat_or_dog_2.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis = 0) 
result = classifier.predict(test_image) 
trainning_set.class_indices 
if result[0][0] == 1: 
    prediction = 'dog' 
else: 
    prediction = 'cat' 
print(prediction) 







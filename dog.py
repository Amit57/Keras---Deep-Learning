#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 23:43:31 2018

@author: amit
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
import os
from os.path import join
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import pandas as pd



image_dir = '/Users/amit/Desktop/Keras/Keras_practice/Classification/Dog/train'      
resize_dir = '/Users/amit/Desktop/Keras/Keras_practice/Classification/Dog/resize'     

labels = pd.read_csv(join(image_dir, 'labels.csv'))
labels.head()

######## Before resizing ##########

pic = labels['id'][3]
img = image.load_img(join(image_dir,pic+".jpg"))
img = image.img_to_array(img)
plt.imshow(img/255.)
plt.show()

###### convert image to grey scale ############
#pic_dir = os.listdir(image_dir)

import glob
pic_dir = [f for f in glob.glob(image_dir+'*.jpg')]


image_rows,image_columns = 200, 200

for pic in pic_dir:
    image_open = Image.open(image_dir+'/'+pic)
    image_size = image_open.resize((image_rows,image_columns))
    image_convert = image_size.convert('L')
    image_convert.save(resize_dir+'/'+pic,"JPEG")

######## Image Flattening ############

gray_dir = os.listdir(resize_dir)
#gray_dir = [f for f in glob.glob(resize_dir+'*.jpg')]

image_matrix = array([array(Image.open(resize_dir + '/' + picture)).flatten()
                for picture in gray_dir],'f')
image_matrix.shape    
####### labeling ########
    
label_type = labels['breed']
label_type[:3]
encoder = LabelEncoder()
encoder.fit(label_type)
encoded_labels = encoder.transform(label_type)
encoded_labels[:5]
get_labels = np_utils.to_categorical(encoded_labels)
get_labels[:10]
get_labels.shape

############### Preparing training data ###############

data,label = shuffle(image_matrix,get_labels,random_state=2)
train_data = [data,label]
print(train_data[0].shape)
print(train_data[1].shape)

############# Modeling ##################


batch_size = 32
nb_classes = 120
nb_epoch = 5
nb_filters = 32
nb_pool = 2
nb_conv = 3

(X,y) = (train_data[0],train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], image_rows, image_columns, 1)
X_test = X_test.reshape(X_test.shape[0], image_rows, image_columns, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])


model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(image_rows, image_columns,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
               verbose=1, validation_data=(X_test, y_test))
            
            
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1, validation_split=0.2)

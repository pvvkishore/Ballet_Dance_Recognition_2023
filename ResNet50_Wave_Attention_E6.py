#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:55:24 2022

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
dataset_path = os.listdir('Cropped Online Dataset')
dance_types = os.listdir('Cropped Online Dataset')
dance_types = sorted(dance_types)
print(dance_types)
#%%
""" Store all the images in a  single list """
import cv2
Dance_list = []
image = []
labels = []
for item in dance_types:
    all_images_class = os.listdir('Cropped Online Dataset'+'/'+item)
    for images in all_images_class:
        img = cv2.imread(str('Cropped Online Dataset'+'/'+item+'/'+images))
        img = cv2.resize(img,[224,224])
        image.append(img)
        labels.append(item)
        Dance_list.append((item,str('Cropped Online Dataset'+'/'+item+'/'+images)))
print(len(Dance_list))
imagea = np.array(image)
imagea = imagea.astype('float32')/255
print(imagea.shape)

#%%
"""Endode the labels"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_labelencoder = LabelEncoder()
Y = y_labelencoder.fit_transform(labels)
Y2 = Y.reshape(-1,1)
enc = OneHotEncoder(handle_unknown='ignore')
Y1 = enc.fit_transform(Y2)
y = Y1.toarray()
print(y.shape)
#%%
"""Find Wavelet transform and Extract Components"""
import pywt
import pywt.data
"""Wavelet Frequency Matrix"""
LH_1 = []
HL_1 = []
HH_1 = []
LH_2 = []
HL_2 = []
HH_2 = []
LH_3 = []
HL_3 = []
HH_3 = []
LH_4 = []
HL_4 = []
HH_4 = []
for i in range(len(image)):
    coeff2 = pywt.dwt2(cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY),'bior1.3')
    LL,(LH,HL,HH) = coeff2
    LL2,(LH2,HL2,HH2) = pywt.dwt2(LL,'bior1.3')
    LL3,(LH3,HL3,HH3) = pywt.dwt2(LL2,'bior1.3')
    LL4,(LH4,HL4,HH4) = pywt.dwt2(LL3,'bior1.3')
    # Level 1 Coefficients
    LH_1.append(LH)
    HL_1.append(HL)
    HH_1.append(HH)
    
    # Level 2 Coefficients
    LH_2.append(LH2)
    HL_2.append(HL2)
    HH_2.append(HH2)
    
    # Level 3 Coefficients
    LH_3.append(LH3)
    HL_3.append(HL3)
    HH_3.append(HH3)
    
    # Level 4 Coefficients
    LH_4.append(LH4)
    HL_4.append(HL4)
    HH_4.append(HH4)
    
LH_1A = np.array(LH_1)
HL_1A = np.array(HL_1)
HH_1A = np.array(HH_1)

LH_2A = np.array(LH_2)
HL_2A = np.array(HL_2)
HH_2A = np.array(HH_2)

LH_3A = np.array(LH_3)
HL_3A = np.array(HL_3)
HH_3A = np.array(HH_3)

LH_4A = np.array(LH_4)
HL_4A = np.array(HL_4)
HH_4A = np.array(HH_4)

print(LH_1A.shape)
print(LH_2A.shape)
print(LH_3A.shape)
print(LH_4A.shape)
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    imagea, y, test_size=0.33, random_state=42)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    LH_1A, y, test_size=0.33, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    HL_1A, y, test_size=0.33, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(
    HH_1A, y, test_size=0.33, random_state=42)

X4_train, X4_test, y4_train, y4_test = train_test_split(
    LH_2A, y, test_size=0.33, random_state=42)
X5_train, X5_test, y5_train, y5_test = train_test_split(
    HL_2A, y, test_size=0.33, random_state=42)
X6_train, X6_test, y6_train, y6_test = train_test_split(
    HH_2A, y, test_size=0.33, random_state=42)

X7_train, X7_test, y7_train, y7_test = train_test_split(
    LH_3A, y, test_size=0.33, random_state=42)
X8_train, X8_test, y8_train, y8_test = train_test_split(
    HL_3A, y, test_size=0.33, random_state=42)
X9_train, X9_test, y9_train, y9_test = train_test_split(
    HH_3A, y, test_size=0.33, random_state=42)

X10_train, X10_test, y10_train, y10_test = train_test_split(
    LH_4A, y, test_size=0.33, random_state=42)
X11_train, X11_test, y11_train, y11_test = train_test_split(
    HL_4A, y, test_size=0.33, random_state=42)
X12_train, X12_test, y12_train, y12_test = train_test_split(
    HH_4A, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X1_train.shape)
print(X2_train.shape)
print(X3_train.shape)
#%%
from keras import layers
from keras.layers import Input, Add, Dense, Multiply, Activation, ZeroPadding2D, BatchNormalization,Flatten,Conv2D,MaxPooling2D,AveragePooling2D
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
# def conv_block(x):
#     x = Conv2D(16,(3,3),activation = 'relu', input_shape = (224,224,3))(x)
#     x = MaxPooling2D(2,2)(x)
#     x = Conv2D(16,(3,3), activation = 'relu')(x)
#     x = MaxPooling2D(2,2)(x)
#     x = Conv2D(16,(3,3), activation = 'relu')(x)
#     x = MaxPooling2D(2,2)(x)
#     x = Conv2D(32,(3,3), activation = 'relu')(x)
#     x = MaxPooling2D(2,2)(x)
#     x = Conv2D(32,(3,3), activation = 'relu')(x)
#     x = MaxPooling2D(2,2)(x)
#     return x
#%% My Model
classes = 8
x_input = Input((224,224,3))
x = ZeroPadding2D((3,3))(x_input)

x = Conv2D(64,(3,3),activation = 'relu', input_shape = (224,224,3))(x)
x = MaxPooling2D(2,2)(x)

x1_input = Input((114,114,1))
x2_input = Input((114,114,1))
x3_input = Input((114,114,1))
x1_shortcut = x1_input
x3_shortcut = x3_input
x1 = Conv2D(64,(1,1),(1,1),padding = 'same')(x1_input)
x2 = Conv2D(64,(1,1),(1,1),padding = 'same')(x2_input)
x3 = Conv2D(64,(1,1),(1,1),padding = 'same')(x3_input)
x12 = Multiply()([x1,x2])
x123 = Multiply()([x3,x12])
x4 = Conv2D(1,(1,1),(1,1),padding = 'same')(x123)
xf = Add()([x4,x1_input])
xf = Add()([xf,x3_input])
#xf = Conv2D(1,(3,3),(1,1),padding = 'valid')(xf)
x = Multiply()([x,xf])
x = Conv2D(64,(3,3), activation = 'relu')(x)
x = MaxPooling2D(2,2)(x)
x = Conv2D(64,(3,3), activation = 'relu')(x)
x = MaxPooling2D(2,2)(x)
x = Conv2D(128,(3,3), activation = 'relu')(x)
x = MaxPooling2D(2,2)(x)
x = Conv2D(128,(3,3), activation = 'relu')(x)
x = MaxPooling2D(2,2)(x)
x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = Dense(classes, activation = 'softmax')(x)

model = Model(inputs = [x_input,x1_input,x2_input,x3_input], 
              outputs=x, name = 'My_DRM')
#%%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(monitor = 'val_loss', patience = 10,
                              verbose = 1, restore_best_weights = True)
reduce1 = ReduceLROnPlateau(monitor = 'val_loss', patience = 10,
                            verbose = 1, factor = 0.5, min_lr = 1e-6)
optimizer = Adam(1e-7)
model.compile(optimizer = optimizer, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

history = model.fit([X_train,X1_train,X2_train,X3_train], 
                    y_train, epochs = 200, 
                    batch_size = 32, verbose=1,
                    validation_data=([X_test,X1_test,X2_test, X3_test],
                                     y_test))
#%% Plot the attention maps
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
#%% Load original image from disk (in openCV format) and resize it
img_path = '/home/user/Desktop/Classical Dance Identification_DL/Cropped Online Dataset/1/Nikolina Nikoleski_ Bho Shambo 01836.jpg'
orig = cv2.imread(img_path)  
resized = cv2.resize(orig, (224,224))
# Load image in tf format       
image = load_img(img_path, target_size=(224,224))
image = img_to_array(image)
image /= 255
image = np.expand_dims(image, axis = 0)
image = imagenet_utils.preprocess_input(image)
image1 = np.expand_dims(LH_1A[84], axis = 0)
image1 = imagenet_utils.preprocess_input(image1)
image2 = np.expand_dims(HL_1A[84], axis = 0)
image2 = imagenet_utils.preprocess_input(image2)
image3 = np.expand_dims(HH_1A[84], axis = 0)
image3 = imagenet_utils.preprocess_input(image3)
image4 = np.expand_dims(LH_2A[84], axis = 0)
image4 = imagenet_utils.preprocess_input(image4)
image5 = np.expand_dims(HL_2A[84], axis = 0)
image5 = imagenet_utils.preprocess_input(image5)
image6 = np.expand_dims(HH_2A[84], axis = 0)
image6 = imagenet_utils.preprocess_input(image6)
# cv2.imshow('img',orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
#%%# Use network to make predictions on the input image and find
# the class label index with the largest probability
preds = model.predict([image,image1,image2,image3])      
i = np.argmax(preds[0])
print(i)
#%%
#import seaborn as sns
import matplotlib.pyplot as plt
train_data = {'loss': history.history['loss'],
              'val_loss': history.history['val_loss'],
              'acc': history.history['accuracy'],
              'val_acc': history.history['val_accuracy']}
#sns.set_style('whitegrid')
fig, (ax1,ax2) = plt.subplots(2, 1, sharex = 'col',
                             figsize = (20, 14))
ax1.plot(history.history['loss'], label = 'Train Loss', linewidth = 4)
ax1.plot(history.history['val_loss'], label = 'validation Loss', linewidth = 4)
ax1.legend(loc = 'best')
ax1.set_title('Loss')
leg = ax1.legend()
leg_lines = leg.get_lines()
leg_text = leg.get_texts()
plt.setp(leg_lines, linewidth = 8)
plt.setp(leg_text, fontsize = 'x-large')
#plt.show()

ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth = 4)
ax2.plot(history.history['val_accuracy'], label='Validation accuracy', linewidth = 4)
ax2.legend(loc='best')
ax2.set_title('Accuracy')
plt.xlabel('Epochs')

leg = ax2.legend()
leg_lines = leg.get_lines()
leg_text = leg.get_texts()
plt.setp(leg_lines, linewidth = 8)
plt.setp(leg_text, fontsize = 'x-large')
plt.show()
#%%
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow
layers_outputs = [layer.output for layer in model.layers[1:]]
#print(layers_outputs)
visualize_model = tensorflow.keras.models.Model(inputs = model.input, 
                                                outputs = layers_outputs)
feature_maps0 = visualize_model.predict([image,image1,image2,image3])
print(len(feature_maps0))
layer_names = [layer.name for layer in model.layers]
print(layer_names)
#%%
# import matplotlib.pyplot as plt
# #get_ipython().run_line_magic('matplotlib', 'inline')
# import cv2
# file = "D:\Dance_Project_2022\results"
# import glob
# for layer_names, feature_maps0 in zip(layer_names, feature_maps0):
#     print(feature_maps0.shape)
#     if len(feature_maps.shape) == 4:
#         channels = feature_maps.shape[-1]
#         size = feature_maps.shape[1]
#         display_grid = np.zeros((size,size*channels))
#         for i in range(channels):
#             x = feature_maps[1][0,:,:,i]
#             x -= x.mean()
#             x /= x.std()
#             x *= 224
#             x += 224
#             x = np.clip(x,0,255).astype('uint8')
#             # tile each filter into a big horizontal grid
#             display_grid[:,i*size:(i+1)*size] = x
            
#             scale = 2./channels
#             plt.figure(figsize=(scale*channels, scale))
#             plt.title(layer_names)
#             plt.grid(False)
#             plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
#         cv2.imwrite('file\messigray{}.png'.format(layer_names), x)
# #%%
# import matplotlib.pyplot as plt
# #get_ipython().run_line_magic('matplotlib', 'inline')
# import cv2
# file = "/home/user/Desktop/Classical Dance Identification_DL\results"
# import glob
# feature_maps = feature_maps0[5] 
# feature_maps = np.array(feature_maps)
# if len(feature_maps.shape) == 4:
#         channels = feature_maps.shape[-1]
#         size = feature_maps.shape[1]
#         display_grid = np.zeros((size,size*channels))
#         for i in range(channels):
#             x = feature_maps[0,:,:,i]
#             x -= x.mean()
#             x /= x.std()
#             x *= 224
#             x += 224
#             x = np.clip(x,0,255).astype('uint8')
#             y = np.clip(x,0,255).astype('uint8')
#             x1 = y
#             # tile each filter into a big horizontal grid
#             display_grid[:,i*size:(i+1)*size] = x
            
#             scale = 2./channels
#             plt.figure(figsize=(scale*channels, scale))
#             plt.title(layer_names)
#             plt.grid(False)
#             plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
#         cv2.imwrite('file\messigray{}.png'.format(layer_names[5]), x1)
#%%
for layer_names, feature_maps0 in zip(layer_names, feature_maps0):
    savedir = {'layer_names': layer_names,
               'feature_maps': feature_maps0}
#%%
import scipy.io as sio
sio.savemat('my_arrays.mat', savedir)

    

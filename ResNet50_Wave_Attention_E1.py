#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 10:06:46 2022

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
# fig = plt.figure(figsize=(12,6))
# titles = ['Approximation','Horizontal',
#           'Vertical', 'Diagonal']
# for i in range(len(image)-795):
#     coeff2 = pywt.dwt2(cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY),'bior1.3')
#     LL,(LH,HL,HH) = coeff2
#     LL2,(LH2,HL2,HH2) = pywt.dwt2(LL,'bior1.3')
#     LL3,(LH3,HL3,HH3) = pywt.dwt2(LL2,'bior1.3')
#     LL4,(LH4,HL4,HH4) = pywt.dwt2(LL3,'bior1.3')
# for i, a in enumerate([LL4,LH4,HL4,HH4]):
#     c = np.array(a, dtype = np.uint8)
#     ax = fig.add_subplot(1,4,i+1)
#     ax.imshow(a, interpolation = "nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize = 20)
#     ax.set_xticks([])
#     ax.set_yticks([])
# fig.tight_layout()
# plt.show()
# #%%
# print(LL.shape)
# print(LL2.shape)
# print(LL3.shape)
# print(LL4.shape)
#%%
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
from keras.optimizers import SGD
#%%
"""Wavelet Attention Layer Design"""
# def wave_attention_block(input_shape = (114,114),classes = 8):
    
#     x1_shortcut = x1
#     x3_shortcut = x3
    
#     x1 = Conv2D(1,(1,1),(1,1),padding = 'valid')(x1)
#     x2 = Conv2D(1,(1,1),(1,1),padding = 'valid')(x2)
#     x3 = Conv2D(1,(1,1),(1,1),padding = 'valid')(x3)
    
#     x12 = Multiply()(x1,x2)
#     x312 = Multiply()(x12,x3)
    
#     x312 = Conv2D(1,(1,1),(1,1),padding = 'valid')(x312)
#     x312 = Add()([x312,x1_shortcut,x3_shortcut])
    
#     return(x312)
#%%
x1_input = Input((114,114,1))
x2_input = Input((114,114,1))
x3_input = Input((114,114,1))
x1_shortcut = x1_input
x3_shortcut = x3_input

x1 = Conv2D(16,(1,1),(1,1),padding = 'valid')(x1_input)
x2 = Conv2D(16,(1,1),(1,1),padding = 'valid')(x2_input)
x3 = Conv2D(16,(1,1),(1,1),padding = 'valid')(x3_input)
x12 = Multiply()([x1,x2])
x123 = Multiply()([x3,x12])
x4 = Conv2D(1,(1,1),(1,1),padding = 'valid')(x123)
xf = Add()([x4,x1_input])
xf = Add()([xf,x3_input])
xf = Conv2D(1,(3,3),(1,1),padding = 'valid')(xf)
flt = Flatten()(xf)
dense1 = Dense(1024, activation='relu')(flt)
dense = Dense(8, activation='softmax',
              kernel_initializer = glorot_uniform(seed=0))(dense1)
model = Model(inputs = [x1_input,x2_input,x3_input], outputs = dense)

sgd = SGD(lr=0.000001, momentum=0.8, decay=1e-4, nesterov=False)
model.compile(optimizer=sgd,loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit([X1_train,X2_train,X3_train] , y_train, batch_size=32,epochs = 15,verbose=1)
























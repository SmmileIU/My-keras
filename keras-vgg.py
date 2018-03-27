#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:22:17 2018

@author: zhao
"""

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.utils import to_categorical
import numpy as np


model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

x_train=np.random.random(size=(100,100,100,3))
y_train=to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)
x_test=np.random.random(size=(10,100,100,3))
y_test=to_categorical(np.random.randint(10,size=(10,1)),num_classes=10)

model.fit(x_train,y_train,batch_size=20,epochs=10)
score=model.evaluate(x_test,y_test,batch_size=1)

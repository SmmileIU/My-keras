#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:54:00 2018

@author: zhao
"""

from keras.models import Sequential
from keras.layers import LSTM,Dense
import numpy as np

model=Sequential()
model.add(LSTM(32,return_sequences=True,input_shape=(8,16)))
model.add(LSTM(32,return_sequences=True)) #returns a sequence of vectors of dimension 32
model.add(LSTM(32)) # return a single vector of dimension 32
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

x_train=np.random.random(size=(1000,8,16))
y_train=np.random.random(size=(1000,10))

x_val=np.random.random(size=(500,8,16))
y_val=np.random.random(size=(500,10))

model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=100,epochs=20)

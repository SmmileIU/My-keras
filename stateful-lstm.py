#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:31:57 2018

@author: zhao
"""

from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.utils import plot_model
import numpy as np

batch_size=30
timesteps=10
dims=16

model=Sequential()
model.add(LSTM(32,
               return_sequences=True,
               stateful=True,#  *
               # Note that we have to provide the full batch_input_shape since the network is statefult
               batch_input_shape=(batch_size,timesteps,dims)))
model.add(LSTM(32,return_sequences=True,stateful=True))
model.add(LSTM(32,stateful=True))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

x_train=np.random.random(size=(batch_size*10,timesteps,dims))
y_train=np.random.random(size=(batch_size*10,10))

x_val=np.random.random(size=(batch_size*5,timesteps,dims))
y_val=np.random.random(size=(batch_size*5,10))

model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=batch_size,epochs=100)
plot_model(model,to_file='model1.png')

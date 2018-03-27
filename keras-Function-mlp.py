#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:34:15 2018

@author: zhao
"""

from keras.models import Model
from keras.layers import Input,Dense

inputs=Input(shape=(256,))
x=Dense(32,activation='relu')(inputs)
x=Dense(64,activation='relu')(x)
pre=Dense(10,activation='softmax')(x)
model=Model(inputs=inputs,outputs=pre)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#make data

model.fit(x_train,y_train,batch_size,epochs)
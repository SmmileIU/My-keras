#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:48:13 2018

@author: zhao
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np

model=Sequential()
model.add(Dense(30,activation='relu',input_shape=(30,)))#must this format or input_dim
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
sgd=SGD(lr=0.01,momentum=0,nesterov=0)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train=np.random.random((1000,30))
y_train=to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
x_test=np.random.random((200,30))
y_test=to_categorical(np.random.randint(10,size=(200,1)),num_classes=10)

model.fit(x_train,y_train,batch_size=30,epochs=120)
score=model.evaluate(x_test,y_test,batch_size=20)


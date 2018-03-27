#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:16:39 2018

@author: zhao
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import plot_model
import numpy as np

model=Sequential()
model.add(Dense(30,activation='relu',input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

x_train=np.random.random((1000,20))
y_train=np.random.randint(2,size=(1000,1))
x_test=np.random.random((200,20))
y_test=np.random.randint(2,size=(200,1))

model.fit(x_train,y_train,batch_size=20,epochs=10,validation_split=0.1)
score=model.evaluate(x_test,y_test,batch_size=20,verbose=0)
plot_model(model,to_file='model.png')





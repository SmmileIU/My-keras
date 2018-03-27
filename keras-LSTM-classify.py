#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:39:01 2018

@author: zhao
"""

from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout

model=Sequential()
model.add(Embedding(input_dim,output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#make data

model.fit(x_train,y_train,batch_size=16,epochs=10)
score=model.evaluate(x_test,y_test,batch_size=10)


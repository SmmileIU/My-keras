#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:26:06 2018

@author: zhao
"""
from keras.models import Model
from keras.layers import Input,Conv2D,Flatten,Dense,concatenate,MaxPooling2D

input=Input(shape=(20,20,3))
x=Conv2D(32,(3,3),padding='same',activation='relu')(input)
x=MaxPooling2D(pool_size=(2,2),padding='same')(x)
x=Flatten()(x)

base_model=Model(inputs=input,outputs=x)

a=Input(shape=(20,20,3))
b=Input(shape=(20,20,3))

a_vis=base_model(a)
b_vis=base_model(b)

mr=concatenate([a_vis,b_vis],axis=-1)
q=Dense(1,activation='sigmoid')(mr)

final_model=Model(inputs=[a,b],outputs=q)


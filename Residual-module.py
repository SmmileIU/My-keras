#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:21:05 2018

@author: zhao
"""

from keras.layers import Input,Conv2D,add

img=Input(shape=(224,224,3))

y=Conv2D(3,(3,3),padding='same')(img)

res=add([img,y])
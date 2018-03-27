#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:25:30 2018

@author: zhao
"""

from keras.models import Model
from keras.layers import Input,LSTM,Dense
from keras.layers import concatenate

tweet_a=Input(shape=(140,256))
tweet_b=Input(shape=(140,256))

share_layer=LSTM(32)

share_a=share_layer(tweet_a)
share_b=share_layer(tweet_b)
m=concatenate([share_a,share_b],axis=-1)
q=Dense(32,activation='relu')(m)
q=Dense(1,activation='sigmoid')(q)
model=Model(inputs=[tweet_a,tweet_b],outputs=q)

model.compile(optimizer='adam',loss='binary_crossentropy')

model.fit(x=[data_a,data_b],labels,batch_size,epochs=10)
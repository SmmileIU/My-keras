#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:45:02 2018

@author: zhao
"""
from keras.models import Model
from keras.layers import Input,Embedding,LSTM,Dense
from keras.layers import concatenate

main_input=Input(shape=(100,),dtype='int32',name='main_input')#  *
x=Embedding(input_dim=10000,output_dim=512,input_length=100)(main_input)# **
x=LSTM(32)(x)
aux_output=Dense(1,activation='sigmoid',name='aux_output')(x)
aux_input=Input(shape=(5,),dtype='int32',name='aux_input')
q=concatenate([x,aux_input])# *
q=Dense(32,activation='relu')(q)
q=Dense(64,activation='relu')(q)
q=Dense(128,activation='relu')(q)
main_output=Dense(1,activation='sigmoid',name='main_output')(q)

model=Model(inputs=[main_input,aux_input],outputs=[main_output,aux_output])#  ***
model.compile(optimizer='adam',
              loss={'main_output':'binary_crossentropy',# *
                    'aux_output':'binary_crossentropy'},
              loss_weights={'main_output':1,# *
                            'aux_output':0.2})

model.fit({'main_input':headline_data,'aux_input':additional_data},# *
          {'main_output':labels,'aux_output':labels},# *(the same labels)
          epochs=50,batch_size=32)
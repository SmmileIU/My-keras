#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:36:44 2018

@author: zhao
"""
# the image-answer-model

from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.layers import Input,Embedding,LSTM
from keras.layers import concatenate

img_model=Sequential()
img_model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(224,224,3)))
img_model.add(Conv2D(32,(3,3),activation='relu'))
img_model.add(MaxPooling2D(pool_size=(2,2)))
img_model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
img_model.add(Conv2D(64,(3,3),activation='relu'))
img_model.add(MaxPooling2D(pool_size=(2,2)))
img_model.add(Flatten())

img_input=Input(shape=(224,224,3))
img_output=img_model(img_input)

ques_input=Input(shape=(100,),dtype='int32')# *
ques=Embedding(input_dim=10000,output_dim=256,input_length=100)(ques)# *
ques_out=LSTM(64)(ques)# *(the same with the number of  the filters of the last conv)

mer=concatenate([img_output,ques_out],axie=-1)
res=Dense(1000,activation='softmax')(mer)

model=Model(inputs=[img_input,ques_input],outputs=res)










# the video-answer-model
from keras.layers import TimeDistributed
video_input=Input(shape=(100,224,224,3))
encoded=TimeDistributed(img_model)(video_input)
encoded_video=LSTM(64)(encoded)

question_model=Model(inputs=ques_input,outputs=ques_out)
question_input=Input(shape=(100,),dtype='int32')
encoded_question=question_model(question)

mer=concatenate([encoded_question,encoded_video],axie=-1)
resu=Dense(1000,activation='softmax')(mer)

final_model=Model(inputs=[video_input,question_input],outputs=resu)

















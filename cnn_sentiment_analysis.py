#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:40:28 2019

@author: roshan
"""
#======================Sentiment Analysis Using CNN ============================================#

#=======Import Necessary libararies=======================#

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re
import string
#=======Hyperparams if GPU is available==========================#

if tf.test.is_gpu_available():
    #===For GPU====================================#
    BATCH_SIZE = 128                          # Number of examples used in each iteration
    EPOCHS = 2                                # Number of passes through entire dataset
    VOCAB_SIZE = 30000                        # Size of vocabulary dictionary
    MAX_LEN = 500                             # Max length of tweet (in words)
    EMBEDDING_DIM = 40                        # Dimension of word embedding vector


#========Hyperparams for CPU training==============================#
else:
    #==For CPU====================#
    BATCH_SIZE = 32
    EPOCHS = 2
    VOCAB_SIZE = 20000
    MAX_LEN = 90
    EMBEDDING_DIM = 40
    
    
#===============Reading dataset================================+#

train_data=pd.read_csv('train.csv')
text=train_data['tweet']
target=train_data['label']    

#==========Splitting dataset into train and validation dataset===============#

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(text,target,stratify=target)

test_data=pd.read_csv('test.csv')
test=test_data['tweet']

#==========Custom Tokenizer===================================================#

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


imdb_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
imdb_tokenizer.fit_on_texts(text.values)

x_train_seq = imdb_tokenizer.texts_to_sequences(X_train.values)
x_val_seq = imdb_tokenizer.texts_to_sequences(X_test.values)
x_test_seq=imdb_tokenizer.texts_to_sequences(test.values)


x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", value=0)
x_val = sequence.pad_sequences(x_val_seq, maxlen=MAX_LEN, padding="post", value=0)
x_test = sequence.pad_sequences(x_test_seq, maxlen=MAX_LEN, padding="post", value=0)

print('First sample before preprocessing: \n', X_train.values[0], '\n')
print('First sample after preprocessing: \n', x_train[0])

#====Algorithms Parameters=====================================================#

NUM_FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 250

#==Building CNN Model===========================================================#

model = Sequential()

#====we start off with an efficient embedding layer which maps our vocab indices into EMBEDDING_DIM dimensions===#
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
model.add(Dropout(0.2))


model.add(Conv1D(NUM_FILTERS,
                 KERNEL_SIZE,
                 padding='valid',
                 activation='relu',
                 strides=1))


model.add(GlobalMaxPooling1D())

model.add(Dense(HIDDEN_DIMS))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#===fit a model====================================================================#
model.fit(x_train, y_train.values,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.1,
          verbose=2)

#===Evaluate the model==============================================================#

score, acc = model.evaluate(x_val, y_test.values, batch_size=BATCH_SIZE)
print('\nAccuracy: ', acc*100)

pred = model.predict_classes(x_test)
prediction=pd.concat([test_data['id'],pd.DataFrame(pred)],axis=1,names=(['id','label']),index=False)
prediction.to_csv('tweet_submision.csv')
























   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

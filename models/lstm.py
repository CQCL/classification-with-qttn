import os
from random import shuffle
from xmlrpc.server import SimpleXMLRPCRequestHandler
os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

import pickle

set_random_seed(812)

# Helper function to transform exact data used in TN experiments to correct form
def preprocess_lstm_input(data, labels, max_words, max_len):
    # data = []
    # for line in X:
    #     new_str = ""
    #     for x in line:
    #         new_str+= f'{x} '
    #     data.append(new_str)
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(data)
    sequences = tok.texts_to_sequences(data)
    input = sequence.pad_sequences(sequences, maxlen=max_len)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = labels.reshape(-1, 1)
    return input, labels

data_name = 'Protein-binding'

load_path = f'./classification/data/PTN/{data_name}/unibox/'

w2i = pickle.load(file=open(f'{load_path}{"w2i"}', 'rb'))
train_data = pickle.load(file=open(f'{load_path}{"train_data"}', 'rb'))
val_data = pickle.load(file=open(f'{load_path}{"val_data"}', 'rb'))
test_data = pickle.load(file=open(f'{load_path}{"test_data"}', 'rb'))
i2w = dict(zip(w2i.values(), w2i.keys()))

# Helper function for SCTN data
def convert_sctn_data(X):
    out_data = []
    out_labels = []
    for item in X:
        for lab in item["labels"]:
            if np.allclose(lab, [1, 0]):
                out_labels.append(0)
            else:
                out_labels.append(1)
        for sent in item["words"]:
            sent_temp = []
            for word in sent:
                sent_temp.append(i2w[word])
            out_data.append(sent_temp)
    return out_data, out_labels

# train_data, train_labels = convert_sctn_data(train_data)
# val_data, val_labels = convert_sctn_data(val_data)
# test_data, test_labels = convert_sctn_data(test_data)

train_labels = [0 if np.allclose(lab, [1, 0]) else 1 for lab in train_data["labels"]]
val_labels = [0 if np.allclose(lab, [1, 0]) else 1 for lab in val_data["labels"]]
test_labels = [0 if np.allclose(lab, [1, 0]) else 1 for lab in test_data["labels"]]
train_data = [[i2w[w] for w in data] for data in train_data["words"]]
val_data = [[i2w[w] for w in data] for data in val_data["words"]]
test_data = [[i2w[w] for w in data] for data in test_data["words"]]
print(len(train_data), len(train_labels))
print(len(val_data), len(val_labels))
print(len(test_data), len(test_labels))

max_len = max(max([len(t) for t in train_data]), max([len(t) for t in val_data]),  max([len(t) for t in test_data]))
max_words = max(w2i.values())+1
print(f'{max_len=}')
print(f'{max_words=}')

train_data, train_labels = preprocess_lstm_input(train_data, train_labels, max_words, max_len)
val_data, val_labels = preprocess_lstm_input(val_data, val_labels, max_words, max_len)
test_data, test_labels = preprocess_lstm_input(test_data, test_labels, max_words, max_len)

# define model
num_embed = 3
num_LSTM = 2 # 1 = 20 + 2, 2 = 48 + 2 (LSTM), 1 = 7 + 2 , 5 = 45 + 6  (RNN)
num_dense = 1

def build_model():
    
    model = Sequential()
    
    # LSTM option
    model.add(Input([max_len]))
    model.add(Embedding(max_words, num_embed))
    model.add(LSTM(num_LSTM, return_sequences=False))

    # RNN option
    # model.add(Input([max_len]))
    # model.add(Embedding(max_words, num_embed))
    # model.add(SimpleRNN(num_LSTM, return_sequences=False))

    # model.add(Dense(num_dense))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

lr = 0.01

checkpoint_filepath = f'./{data_name}/{lr}/{num_LSTM}/checkpoint.model.keras'
# print(f'RNN, smal vocab {data_name=}')
print(f'LSTM, AdamW, {data_name=}')
print(f'{lr=}')
vals = []

model = build_model()
model.summary()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.AdamW(lr), metrics=['accuracy'])

model_checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True)

# call
model.fit(train_data, train_labels, batch_size=64,epochs=30,
        validation_data=[val_data, val_labels], callbacks=[model_checkpoint_callback])

# eval on best model (according to val set)
best_model = load_model(checkpoint_filepath)

# evaluate on test
accr = best_model.evaluate(test_data, test_labels)
print(accr)
vals.append(accr[1])
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import time

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

#Keras
import keras
from keras import models
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint

import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

import warnings
warnings.filterwarnings('ignore')

val_csv_file = './extract/data_full_features.csv'

data = pd.read_csv('data_train_val_full_features.csv')
#data = pd.read_csv('data.csv')
# data.head()
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
# print(y)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

print(X.shape)

# read test data
val_data = pd.read_csv(val_csv_file)
val_data.head()
val_genre_list = val_data.iloc[:, -1]
y_val = encoder.fit_transform(val_genre_list)
X_val = scaler.fit_transform(np.array(val_data.iloc[:, :-1], dtype = float))
print(X_val.shape)

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
highest_acc = 0
best_model = None
ite = 0
millis = str(int(round(time.time() * 1000)))
os.mkdir("./model_nn/" + millis)
model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(X.shape[1],))) 
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

filepath="./model_nn/" + millis + "/lstm-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(X, y, epochs=200, batch_size=128, validation_data=(X_val, y_val), callbacks=callbacks_list)

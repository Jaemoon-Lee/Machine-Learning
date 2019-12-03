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
from keras.models import load_model

import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

import warnings
warnings.filterwarnings('ignore')

val_csv_file = './extract/data_full_features.csv'

best_model = load_model("./model_nn/1556814383795/lstm-improvement-14-0.77.hdf5")

# read test data
val_data = pd.read_csv(val_csv_file)
# print(val_data.shape)
# val_data.head()
val_genre_list = val_data.iloc[:, -1]
encoder = LabelEncoder()
y_val = encoder.fit_transform(val_genre_list)

scaler = StandardScaler()
X_val = scaler.fit_transform(np.array(val_data.iloc[:, :-1], dtype = float))

print(X_val.shape)

test_loss, test_acc = best_model.evaluate(X_val,y_val)

count_prog = count_nonprog = 0
acc_prog = acc_nonprog = 0
results = best_model.predict(X_val)
for i,re in enumerate(results):
    if y_val[i] == 0:
        count_nonprog = count_nonprog + 1
        if re[0] > re[1]:
            acc_nonprog = acc_nonprog + 1
    if y_val[i] == 1:
        count_prog = count_prog + 1
        if re[0] <= re[1]:
            acc_prog = acc_prog + 1

print(f'acc prog {acc_prog} / {count_prog} - nonprog {acc_nonprog} / {count_nonprog}')
print('test_acc: ',test_acc)

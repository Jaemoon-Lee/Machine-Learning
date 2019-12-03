# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')


def extract_features(filename, genre): # return text to add into csv
    y, sr = librosa.load(songname, mono=True, duration=600)
    print(y.shape)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    poly_features = librosa.feature.poly_features(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(y=y,sr=sr)
    # append mean
    to_append = f'{np.mean(chroma_stft)} {np.mean(chroma_cqt)} {np.mean(chroma_cens)} {np.mean(rmse)} '
    to_append += f'{np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(contrast)} {np.mean(flatness)} {np.mean(zcr)} '    
    to_append += f'{np.mean(melspectrogram)} {np.mean(poly_features[0])} {np.mean(poly_features[1])}'
    for e in tonnetz:
        to_append += f' {np.mean(e)}'
    for e in tempogram:
        to_append += f' {np.mean(e)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    # append var
    to_append = f'{np.var(chroma_stft)} {np.var(chroma_cqt)} {np.var(chroma_cens)} {np.var(rmse)} '
    to_append += f'{np.var(spec_cent)} {np.var(spec_bw)} {np.var(rolloff)} {np.var(contrast)} {np.var(flatness)} {np.var(zcr)} '    
    to_append += f'{np.var(melspectrogram)} {np.var(poly_features[0])} {np.var(poly_features[1])}'
    for e in tonnetz:
        to_append += f' {np.var(e)}'
    for e in tempogram:
        to_append += f' {np.var(e)}'
    for e in mfcc:
        to_append += f' {np.var(e)}'

    # label
    to_append += f' {genre}'
    return to_append

header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

filepath = './path.txt'
extract_folder = './extract'
if not os.path.exists(extract_folder):
    os.mkdir(extract_folder)
csv_file = './extract/data_full_features4.csv'

prog_path = nonprog_path = None
with open(filepath) as fp:  
   prog_path1 = fp.readline().strip()
   nonprog_path1 = fp.readline().strip()
   prog_paths = np.array([prog_path1])
   nonprog_paths = np.array([nonprog_path1])

file = open(csv_file, 'w', newline='')
genres = 'prog nonprog'.split()
with file:
    writer = csv.writer(file)
    writer.writerow(header)

# for prog_path in prog_paths:
#     for progfile in os.listdir(prog_path):
#         songname = f'{prog_path}/{progfile}'
#         to_append = extract_features(songname, 'prog')
#         file = open(csv_file, 'a', newline='')
#         with file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())

for nonprog_path in nonprog_paths:
    for non_prog_file in os.listdir(nonprog_path):
        songname = f'{nonprog_path}/{non_prog_file}'
        to_append = extract_features(songname, 'nonprog')
        file = open(csv_file, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
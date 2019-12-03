################################################################################
################################################################################
import keras
# Keras
import tensorflow as tf
#from tensorflow.python import keras
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers

from sklearn.preprocessing import LabelEncoder, StandardScaler

from featureExtractTool import *

import numpy as np
from numpy import newaxis
from collections import Counter

config = tf.ConfigProto(device_count={"CPU": 6})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# Define pattern size
NUM_OF_SAMPLE_PER_WINDOW = 160
NUM_OF_OVERLAP_SAMPLE = 80

# Set prog songs
PATH_TO_PROG_SONGS_VALIDATION = "../validation songs/prog"
PROG_LABEL = "prog"

# Set nonprog songs
PATH_TO_NONPROG_SONGS_VALIDATION = "../validation songs/nonprog"
NONPROG_LABEL = "nonprog"







def classify_pattern(a_Model, a_Pattern):
    t_PtPredFracArr = a_Model.predict(a_Pattern)
    return t_PtPredFracArr[0,0]

def classify_song(a_Model, a_Patterns, a_Label):

    # Init dict to store results
    t_Dict = {"#Songs" : 1, "#CorrectSongPred" : 0, "#Patterns" : 0, "#CorrectPatternPred" : 0}

    # Record number of patterns
    t_Dict["#Patterns"] = a_Patterns.shape[0]
    t_SumOfPredFrac = 0

    # Classify each pattern in this song
    for t_Pt in a_Patterns:

        # resize 2d pattern into 3D
        t_PtPredFrac = classify_pattern(a_Model, t_Pt[newaxis,:,:])
        #print("t_PtPredFrac=", t_PtPredFrac)

        t_SumOfPredFrac += t_PtPredFrac

        # Round predicted result to 1 or 0
        t_PtPred = int(round(t_PtPredFrac))

        # Record correct prediction
        if(t_PtPred==a_Label):
            t_Dict["#CorrectPatternPred"] += 1

    # Majority voting
    if(t_Dict["#CorrectPatternPred"] > t_Dict["#Patterns"]/2):
        print("avg = ",t_SumOfPredFrac/a_Patterns.shape[0])
    #if(t_SumOfPredFrac/a_Patterns.shape[0] < 0.5):
        t_Dict["#CorrectSongPred"] = 1

    return t_Dict



def test_model(a_Model, a_PathToSongDir, a_Label, a_SizeOfWindow, a_NumOfOverlapSample):

    # Init recorder of classification results
    t_AllResultsDict={}


    # Get name of each song and save as list
    t_AllFiles = os.listdir(a_PathToSongDir)

    # Test one song each time
    for i, t_SongName in enumerate(t_AllFiles):
        t_PathToSingleSong = f'{a_PathToSongDir}/{t_SongName}'
        t_Patterns, t_Labels = generate_patterns_for_one_song(t_PathToSingleSong,
                                    a_Label, a_SizeOfWindow, a_NumOfOverlapSample)

        for j in range(t_Patterns.shape[2]):
            t_Scaler = StandardScaler()
            t_Patterns[:, j, :] = t_Scaler.fit_transform(t_Patterns[:, j, :])

        t_DictSingleSong = classify_song(a_Model, t_Patterns, a_Label)
        t_AllResultsDict = Counter(t_AllResultsDict) + Counter(t_DictSingleSong)

        print("Validating song No.", i+1, "/", len(t_AllFiles), "....Current results ", t_AllResultsDict)







t_Model = tf.keras.models.load_model('LSTM_128_32_new.h5')

t_Encoder = LabelEncoder()
t_Labels = t_Encoder.fit_transform([PROG_LABEL, NONPROG_LABEL])
print("Label = ", t_Labels)

#test_model(t_Model, PATH_TO_NONPROG_SONGS_VALIDATION, 1, NUM_OF_SAMPLE_PER_WINDOW, NUM_OF_OVERLAP_SAMPLE)
test_model(t_Model, PATH_TO_PROG_SONGS_VALIDATION, 0, NUM_OF_SAMPLE_PER_WINDOW, NUM_OF_OVERLAP_SAMPLE)

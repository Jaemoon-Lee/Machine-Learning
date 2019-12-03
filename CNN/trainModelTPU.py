################################################################################
# Import the prog and nonprog patterns (pkl file) and train the model

NUM_OF_K_FOLD = 14

################################################################################

import os
import time
import pickle
import warnings
import numpy as np
from numpy import newaxis

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

#Keras
from tensorflow.python import keras
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import regularizers

#from tensorflow.python.keras.layers.recurrent import LSTM



import numpy
seed = 7
numpy.random.seed(seed)

import tensorflow as tf

def preprocess_traing_data():

    # Init Data: X and Label: y
    X = np.array([])
    y = []

    # Find all pkl file in the current folder
    for t_File in os.listdir("./"):
        if t_File.endswith(".pkl"):
            with open('/content/drive/My Drive/Colab Notebooks/'+t_File, "rb") as t_PKLFile:
                t_Dict = pickle.load(t_PKLFile)
                X = np.vstack([X, t_Dict["Patterns"]]) if X.size else t_Dict["Patterns"]
                y = np.concatenate((y, t_Dict["Labels"]), axis=None)
                print("shape of X =", t_Dict["Patterns"].shape, "   y =", len(t_Dict["Labels"]))
                print("Preprocess", t_File, "done!")
    print("Training sets are combined!")


    # Normalize X in each feature space
    t_Scalers = {}
    for i in range(X.shape[2]):
        t_Scalers[i] = StandardScaler()
        X[:, :, i] = t_Scalers[i].fit_transform(X[:, :, i])
    print("Patterns normalization done!")

    # Save scaler for validation set
    millis = int(round(time.time() * 1000))
    t_ScalerFile = f'/content/drive/My Drive/Colab Notebooks/scaler{millis}.sav'
    pickle.dump(t_Scalers, open(t_ScalerFile, 'wb'))
    print("Scaler data saved!")



    # Normalize labels
    t_Encoder = LabelEncoder()
    y = t_Encoder.fit_transform(y)
    print("Labels normalization done!")

    return X, y


def train_model(a_Patterns, a_Labels, a_KSplit):


    # Reshape training data to 4D Depth, width, height, num of patterns
    a_Patterns = a_Patterns[:,:,:,newaxis]


    # Set up K Fold Validation
    t_KFold = StratifiedKFold(n_splits=a_KSplit, shuffle=True, random_state=seed)

    # Start training by K times and save the model with highest ACC
    t_BestACC = 0
    t_AvgAcc = 0
    t_BestModel = None
    for trainIdx, testIdx in t_KFold.split(a_Patterns, a_Labels):

        # Init empty model
        t_Model = models.Sequential()

        # Define input layer
        t_Model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=a_Patterns[0].shape))
        #t_Model.add(ZeroPadding2D(2))


        # Define hidden layers
        t_Model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))


        #t_Model.add(ZeroPadding2D(2))


        t_Model.add(GlobalMaxPooling2D())



        t_Model.add(Flatten(data_format=None))


        t_Model.add(Dense(32))


        t_Model.add(Dropout(0.5, noise_shape=None, seed=None))




        # Define output layer
        t_Model.add(Dense(2, activation='softmax'))

        for layer in t_Model.layers:
          print(layer.output_shape)



        opt = tf.train.AdamOptimizer(0.002)

        t_Model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

        t_Resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
        t_Strategy = tf.contrib.tpu.TPUDistributionStrategy(t_Resolver)
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(t_Model, strategy=t_Strategy)


        tpu_model.fit(a_Patterns[trainIdx], a_Labels[trainIdx], epochs=25, batch_size=1024, validation_data=(a_Patterns[testIdx], a_Labels[testIdx]))
        scores = tpu_model.evaluate(a_Patterns[testIdx], a_Labels[testIdx], verbose=0)

        # Update best model
        print("%s: %.2f%%" % (tpu_model.metrics_names[1], scores[1]*100))
        t_AvgAcc += scores[1];
        if scores[1] > t_BestACC:
            t_BestACC = scores[1]
            t_BestModel = tpu_model
        break


    print('BestACC =', t_BestACC)
    print('AvgAcc =', t_AvgAcc/a_KSplit)


    millis = int(round(time.time() * 1000))
    t_BestModel.save(f'/content/drive/My Drive/Colab Notebooks/CNN_{millis}.h5')






warnings.filterwarnings('ignore')
X, y = preprocess_traing_data()

#X = np.ones((1000, 43, 128))
#y = np.ones((1000))

print("shape of X", X.shape)
print("shape of y", y.shape)
train_model(X, y, 14)

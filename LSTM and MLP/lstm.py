# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import os
import time
# fix random seed for reproducibility
np.random.seed(7)

# ---------- data paths 2000
prog_train_data_files = ["./data/data_0_0.npy","./data/data_0_1.npy","./data/data_0_2.npy","./data/data_0_3.npy","./data/data_0_4.npy"]
nonprog_train_data_files = ["./data/data_1_0.npy", "./data/data_1_1.npy", "./data/data_1_2.npy",
                            "./data/data_2_0.npy", "./data/data_2_1.npy", "./data/data_2_2.npy",
                            "./data/data_3_0.npy", "./data/data_3_1.npy", "./data/data_3_2.npy",
                            "./data/data_4_0.npy", "./data/data_4_1.npy", "./data/data_4_2.npy", "./data/data_4_3.npy"]

prog_val_data_files = ["./data/dataVal_0_0.npy", "./data/dataVal_0_1.npy"]
nonprog_val_data_files = ["./data/dataVal_1_0.npy", "./data/dataVal_1_1.npy", "./data/dataVal_1_2.npy"]

# ----------- data paths 1000
# prog_train_data_files = ["./data/data1000/dataTrain_0_0.npy", "./data/data1000/dataTrain_0_1.npy", "./data/data1000/dataTrain_0_2.npy", 
#                         "./data/data1000/dataTrain_0_3.npy", "./data/data1000/dataTrain_0_4.npy",
#                         "./data/data1000/dataTrain_1_0.npy","./data/data1000/dataTrain_1_1.npy","./data/data1000/dataTrain_1_2.npy",
#                         "./data/data1000/dataTrain_1_3.npy","./data/data1000/dataTrain_1_4.npy"]
# nonprog_train_data_files = ["./data/data1000/dataTrain_2_0.npy", "./data/data1000/dataTrain_2_1.npy", "./data/data1000/dataTrain_2_2.npy",
#                             "./data/data1000/dataTrain_2_3.npy", "./data/data1000/dataTrain_2_4.npy", "./data/data1000/dataTrain_2_5.npy",
#                             "./data/data1000/dataTrain_3_0.npy", "./data/data1000/dataTrain_3_1.npy", "./data/data1000/dataTrain_3_2.npy",
#                             "./data/data1000/dataTrain_3_3.npy", "./data/data1000/dataTrain_3_4.npy", "./data/data1000/dataTrain_3_5.npy",
#                             "./data/data1000/dataTrain_4_0.npy", "./data/data1000/dataTrain_4_1.npy", "./data/data1000/dataTrain_4_2.npy",
#                             "./data/data1000/dataTrain_4_3.npy", "./data/data1000/dataTrain_4_4.npy",
#                             "./data/data1000/dataTrain_5_0.npy", "./data/data1000/dataTrain_5_1.npy", "./data/data1000/dataTrain_5_2.npy",
#                             "./data/data1000/dataTrain_5_3.npy", "./data/data1000/dataTrain_5_4.npy", "./data/data1000/dataTrain_5_5.npy",
#                             "./data/data1000/dataTrain_5_6.npy"]

# prog_val_data_files = ["./data/data1000/dataVal_0_0.npy", "./data/data1000/dataVal_0_1.npy", "./data/data1000/dataVal_0_2.npy"]
# nonprog_val_data_files = ["./data/data1000/dataVal_1_0.npy", "./data/data1000/dataVal_1_1.npy", "./data/data1000/dataVal_1_2.npy",
#                             "./data/data1000/dataVal_1_3.npy", "./data/data1000/dataVal_1_4.npy", "./data/data1000/dataVal_1_5.npy"]

millis = str(int(round(time.time() * 1000)))
os.mkdir("./model/" + millis)

class DataGenerator(object):
    def __init__(self, batch_size, prog_files, nonprog_files):
        self.train_X = np.array([])
        self.train_Y = np.array([])
        self.batch_size = batch_size
        self.prog_files = prog_files
        self.nonprog_files = nonprog_files
        self.current_idx = 0
        self.data_length = 0
        self.current_prog_file_idx = 0
        self.current_nonprog_file_idx = 0
        self.nonprog_per_prog = 2 # ratio of size of trainning data of nonprog / prog

    def generate(self):
        while True:
            if self.train_X.shape[0] == 0:
                self.nonprog_per_prog = np.random.randint(3)
                self.train_X = np.load(self.prog_files[self.current_prog_file_idx])
                self.train_Y = np.ones(self.train_X.shape[0])
                
                for j in range(self.nonprog_per_prog):
                    self.train_X = np.append(self.train_X, np.load(self.nonprog_files[(self.current_nonprog_file_idx + j) % len(self.nonprog_files)]), axis=0)
                
                self.train_Y = np.append(self.train_Y, np.zeros(self.train_X.shape[0] - self.train_Y.shape[0]))
                self.current_idx = 0
                self.data_length = self.train_X.shape[0]
                self.train_X, self.train_Y = shuffle(self.train_X, self.train_Y)
            
            e = min(self.current_idx + self.batch_size, self.data_length)
            x = self.train_X[self.current_idx : e]
            y = self.train_Y[self.current_idx : e]

            self.current_idx = min(self.current_idx + self.batch_size, self.data_length)
            if self.current_idx == self.data_length:
                self.current_idx = 0
                self.train_X = np.array([])
                self.train_Y = np.array([])
                self.data_length = 0
                self.current_prog_file_idx = (self.current_prog_file_idx + 1) % len(self.prog_files)
                self.current_nonprog_file_idx = (self.current_nonprog_file_idx + self.nonprog_per_prog) % len(self.nonprog_files) 

            yield x, y

opt = Adam()

hidden_size = 205
num_steps = 2000 # length of segment of each sample
train_data_size = 5000
val_data_size = 4000

train_batch_size = 128
val_batch_size = 92
nb_epochs = 1000000

steps_per_epoch = int(train_data_size / train_batch_size)
validation_steps = int(val_data_size / val_batch_size)

input_shape = (num_steps, hidden_size)

print('Build LSTM RNN model ...')
model = Sequential()
# model.add(LSTM(units=train_X.shape[2], dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=hidden_size, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=hidden_size, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=2, activation='softmax'))

print("Compiling ...")
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

# --- check point with val_acc
# filepath="./model/" + millis + "/lstm-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

# --- checkpoint with acc
filepath="./model/" + millis + "/lstm-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1)

callbacks_list = [checkpoint]

print("Training ...")
train_data_generator = DataGenerator(train_batch_size, prog_train_data_files, nonprog_train_data_files)
val_data_generator = DataGenerator(val_batch_size, prog_val_data_files, nonprog_val_data_files)

# ------- training with validation set
# model.fit_generator(train_data_generator.generate(), validation_data=val_data_generator.generate(), validation_steps=validation_steps, 
#                     epochs=nb_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks_list)

# ------- training without validation set
model.fit_generator(train_data_generator.generate(), epochs=nb_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks_list)

# save model
model.save("./model/" + millis + "/lstm_model.h5")
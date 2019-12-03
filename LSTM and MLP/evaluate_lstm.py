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
from keras.models import load_model
import os
import time
# fix random seed for reproducibility
np.random.seed(7)

#-------------------------- dataset 2000
# +++++++ validate on test set
# prog_val_data_file_paths = [["./data/dataVal_0_0.npy", "./data/dataVal_0_1.npy"]]
# prog_val_seg_file_paths = ["./data/dataVal_seg_0.npy"]
# nonprog_val_data_file_paths = [["./data/dataVal_1_0.npy", "./data/dataVal_1_1.npy", "./data/dataVal_1_2.npy"]]
# nonprog_val_seg_file_paths = ["./data/dataVal_seg_1.npy"]

# +++++++ validate on full validation set
# prog_val_data_file_paths = [["./data/data2000-Full/dataVal_0_0.npy", "./data/data2000-Full/dataVal_0_1.npy", "./data/data2000-Full/dataVal_0_2.npy"]]
# prog_val_seg_file_paths = ["./data/data2000-Full/dataVal_seg_0.npy"]
# nonprog_val_data_file_paths = [["./data/data2000-Full/dataVal_1_0.npy", "./data/data2000-Full/dataVal_1_1.npy", "./data/data2000-Full/dataVal_1_2.npy"]]
# nonprog_val_seg_file_paths = ["./data/data2000-Full/dataVal_seg_1.npy"]

# # +++++++ validate on newest-full validation set
# prog_val_data_file_paths = [["./data2000-Full-New/dataVal_0_0.npy", "./data2000-Full-New/dataVal_0_1.npy", "./data2000-Full-New/dataVal_0_2.npy"]]
# prog_val_seg_file_paths = ["./data2000-Full-New/dataVal_seg_0.npy"]
# nonprog_val_data_file_paths = [["./data2000-Full-New/dataVal_1_0.npy", "./data2000-Full-New/dataVal_1_1.npy", "./data2000-Full-New/dataVal_1_2.npy"]]
# nonprog_val_seg_file_paths = ["./data2000-Full-New/dataVal_seg_1.npy"]

# +++++++ validate on test set
prog_val_data_file_paths = [["./data/data2000-Full-New/dataTest_full_0_0.npy", "./data/data2000-Full-New/dataTest_full_0_1.npy", "./data/data2000-Full-New/dataTest_full_0_2.npy", "./data/data2000-Full-New/dataTest_full_0_3.npy"],
                            ["./data/data2000-Full-New/dataTest_full_1_0.npy", "./data/data2000-Full-New/dataTest_full_1_1.npy", "./data/data2000-Full-New/dataTest_full_1_2.npy"],
                            ["./data/data2000-Full-New/dataTest_full_2_0.npy", "./data/data2000-Full-New/dataTest_full_2_1.npy", "./data/data2000-Full-New/dataTest_full_2_2.npy", "./data/data2000-Full-New/dataTest_full_2_3.npy"]]
prog_val_seg_file_paths = ["./data/data2000-Full-New/dataTest_full_seg_0.npy", "./data/data2000-Full-New/dataTest_full_seg_1.npy", "./data/data2000-Full-New/dataTest_full_seg_2.npy"]
nonprog_val_data_file_paths = [["./data/data2000-Full-New/dataTest_full_3_0.npy", "./data/data2000-Full-New/dataTest_full_3_1.npy"],
                           ["./data/data2000-Full-New/dataTest_full_4_0.npy", "./data/data2000-Full-New/dataTest_full_4_1.npy"],
                           ["./data/data2000-Full-New/dataTest_full_5_0.npy", "./data/data2000-Full-New/dataTest_full_5_1.npy"]]
nonprog_val_seg_file_paths = ["./data/data2000-Full-New/dataTest_full_seg_3.npy", "./data/data2000-Full-New/dataTest_full_seg_4.npy", "./data/data2000-Full-New/dataTest_full_seg_5.npy"]


# +++++++ validate on training set
# prog_val_data_file_paths = [["./data/dataTrain_0_0.npy", "./data/dataTrain_0_1.npy", "./data/dataTrain_0_2.npy", "./data/dataTrain_0_3.npy", "./data/dataTrain_0_4.npy"]]
# prog_val_seg_file_paths = ["./data/dataTrain_seg_0.npy"]
# nonprog_val_data_file_paths = [["./data/dataTrain_1_0.npy", "./data/dataTrain_1_1.npy", "./data/dataTrain_1_2.npy"],
#                                 ["./data/dataTrain_2_0.npy", "./data/dataTrain_2_1.npy", "./data/dataTrain_2_2.npy"],
#                                 ["./data/dataTrain_3_0.npy", "./data/dataTrain_3_1.npy", "./data/dataTrain_3_2.npy"],
#                                 ["./data/dataTrain_4_0.npy", "./data/dataTrain_4_1.npy", "./data/dataTrain_4_2.npy", "./data/dataTrain_4_3.npy"]]
# nonprog_val_seg_file_paths = ["./data/dataTrain_seg_1.npy", "./data/dataTrain_seg_2.npy", "./data/dataTrain_seg_3.npy", "./data/dataTrain_seg_4.npy"]

# load model 2000
model = load_model('./model/lstm-improvement-390-0.77.hdf5')

# model = load_model('./model/lstm-improvement-419-0.80.hdf5')


#------------------------------dataset 1000

# +++++++ validate on test set
# prog_val_data_file_paths = [["./data/data1000/dataVal_0_0.npy", "./data/data1000/dataVal_0_1.npy", "./data/data1000/dataVal_0_2.npy"]]
# prog_val_seg_file_paths = ["./data/data1000/dataVal_seg_0.npy"]
# nonprog_val_data_file_paths = [["./data/data1000/dataVal_1_0.npy", "./data/data1000/dataVal_1_1.npy", "./data/data1000/dataVal_1_2.npy",
#                             "./data/data1000/dataVal_1_3.npy", "./data/data1000/dataVal_1_4.npy", "./data/data1000/dataVal_1_5.npy"]]
# nonprog_val_seg_file_paths = ["./data/data1000/dataVal_seg_1.npy"]

# load model 1000
# model = load_model('./model/model1000/BEST-lstm-improvement-67-0.73.hdf5')

#------------------------------test songs
prog_ratio = 0.4
print("\nPredict whole songs ....")

# load val data
val_X_prog = np.array([])
for j, prog_val_data_files in enumerate(prog_val_data_file_paths):
    for prog_val_data_file in prog_val_data_files:
        if val_X_prog.shape[0] == 0:
            val_X_prog = np.load(prog_val_data_file)
        else:
            val_X_prog = np.append(val_X_prog, np.load(prog_val_data_file), axis=0)
    val_Y_prog = np.ones(val_X_prog.shape[0])
    val_seg_prog = np.load(prog_val_seg_file_paths[j])
    acc_prog = 0
    acc_prog_2 = 0 # count with mean output
    no_prog = 0

    # fix error in seg
    l = val_seg_prog.shape[0]
    for i in range(l - 1):
        if val_seg_prog[i] > val_seg_prog[i+1]:
            e = i + 1
            for j in range(i + 1,l - 1):
                e = j
                if val_seg_prog[j] > val_seg_prog[j+1]:
                    break
            if e == l - 2 and val_seg_prog[e] < val_seg_prog[e + 1]:
                e += 1
            for j in range(i+1, e+1):
                val_seg_prog[j] += val_seg_prog[i]
    
    print(val_seg_prog)

    for i,seg in enumerate(val_seg_prog):
        # print(f"prog song: {i} / {val_seg_prog.shape[0]}")
        s = 0
        if i > 0:
            s = int(val_seg_prog[i - 1])
        if s < int(seg):
            no_prog = no_prog + 1
            X = val_X_prog[s:int(seg)]
            results = model.predict(X)
            # print(results)
            count = 0
            for re in results:
                if re[0] <= re[1]:
                    count = count + 1

            re0 = np.mean(results.T[0])
            re1 = np.mean(results.T[1])
            
            print(f"prog song {i} vote: {count} / {len(results)} , mean {re0} {re1}")
            
            if count >= (len(results) * prog_ratio):
                acc_prog = acc_prog + 1

            if re1 >= prog_ratio:
                acc_prog_2 = acc_prog_2 + 1
    print("--------------Prog acc: ", acc_prog, acc_prog_2, no_prog)
    val_X_prog = np.array([])

val_X_nonprog = np.array([])
for j,nonprog_val_data_files in enumerate(nonprog_val_data_file_paths): 
    for nonprog_val_data_file in nonprog_val_data_files:
        if val_X_nonprog.shape[0] == 0:
            val_X_nonprog = np.load(nonprog_val_data_file)
        else:
            val_X_nonprog = np.append(val_X_nonprog, np.load(nonprog_val_data_file), axis=0)
    val_Y_nonprog = np.zeros(val_X_nonprog.shape[0])
    val_seg_nonprog = np.load(nonprog_val_seg_file_paths[j])

    acc_nonprog = 0
    acc_nonprog_2 = 0
    no_nonprog = 0

    # fix error in seg
    # fix error in seg
    l = val_seg_nonprog.shape[0]
    for i in range(l - 1):
        if val_seg_nonprog[i] > val_seg_nonprog[i+1]:
            e = i + 1
            for j in range(i + 1,l - 1):
                e = j
                if val_seg_nonprog[j] > val_seg_nonprog[j+1]:
                    break
            if e == l - 2 and val_seg_nonprog[e] < val_seg_nonprog[e + 1]:
                e += 1
            for j in range(i+1, e+1):
                val_seg_nonprog[j] += val_seg_nonprog[i]
    if val_seg_nonprog[l - 1] < val_seg_nonprog[l - 2]:
        val_seg_nonprog[l - 1] += val_seg_nonprog[l - 2]

    print(val_seg_nonprog)

    for i,seg in enumerate(val_seg_nonprog):
        s = 0
        if i > 0:
            s = int(val_seg_nonprog[i - 1])
        if s < int(seg):
            no_nonprog = no_nonprog + 1
            X = val_X_nonprog[s:int(seg)]
            results = model.predict(X)
            count = 0
            for re in results:
                if re[0] > re[1]:
                    count = count + 1

            re0 = np.mean(results.T[0])
            re1 = np.mean(results.T[1])
            
            print(f"nonprog song {i} vote: {count} / {len(results)}, mean {re0} {re1}")

            if count > (len(results) * (1 - prog_ratio)):
                acc_nonprog = acc_nonprog + 1

            if re0 > (1 - prog_ratio):
                acc_nonprog_2 = acc_nonprog_2 + 1
    val_X_nonprog = np.array([])
    print("-------------------Nonprog acc: ", acc_nonprog, acc_nonprog_2, no_nonprog)
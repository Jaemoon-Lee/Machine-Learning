import numpy as np
import librosa
import math
import re
import os
import sys

class GenreFeatureDataVal:   
    # # extract old val data
    # file_path = ['E:\\MachineLearning\\Validation_Set\\Prog',
    #             'E:\\MachineLearning\\Validation_Set\\Non-Prog']
    # preprocessed_data = 'data\\data2000-Full\\dataVal'

    # extract new val data
    # file_path = ['E:\\MachineLearning\\new_validation_set\\Validation_Set\\Prog',
    #             'E:\\MachineLearning\\new_validation_set\\Validation_Set\\Non-Prog']
    # preprocessed_data = 'data\\data2000-Full-New\\dataVal'

    # extract train data
    # file_path = ['E:\\MachineLearning\\prog\\prog', 
    #             'E:\\MachineLearning\\prog\\prog1', 
    #             'E:\\MachineLearning\\nonprog\\folder',
    #             'E:\\MachineLearning\\nonprog\\folder1',
    #             'E:\\MachineLearning\\nonprog\\folder2',
    #             'E:\\MachineLearning\\nonprog\\folder3',]
    # preprocessed_data = 'data\\data1000\\dataTrain'

    # extract test data
    file_path = ['E:/MachineLearning/full_test_set/prog',
                'E:/MachineLearning/full_test_set/prog1',
                'E:/MachineLearning/full_test_set/prog2',
                'E:/MachineLearning/full_test_set/non_prog',
                'E:/MachineLearning/full_test_set/non_prog1',
                'E:/MachineLearning/full_test_set/non_prog2']
    preprocessed_data = './data/data2000-Full-New/dataTest_full'

    train_X = train_Y = None
    test_X = test_Y = None

    def __init__(self, normalize, full_data):
        self.normalize = normalize
        self.full_data = full_data

    def load_and_save_data(self, path_id):
        files = self.path_to_audiofiles(self.file_path[path_id])
        self.train_X, seg_data = self.extract_features_files(files, f'{self.preprocessed_data}_{path_id}')
        # with open(f'{self.preprocessed_data}_{path_id}.npy', 'wb') as f:
        #     np.save(f, self.train_X)
        
        with open(f'{self.preprocessed_data}_seg_{path_id}.npy', 'wb') as f:
            np.save(f, seg_data)
        
    # def normalize(self, arr): # arr is 2D
    #     if self.normalize:
    #         for i in range(arr.shape[0]):
    #             u = max(arr[i,:])
    #             l = min(arr[i,:])
    #             w = u - l
    #             if w == 0:
    #                 arr[i,:] = np.array([1])
    #             arr[i,:] = np.array([(e - l)]) 
    #     else:
    #         return arr


    def extract_feature(self, file, genre, sample_length, step = 1000):
        return_data_x = np.array([])
        return_data_y = np.array([])
        y, sr = librosa.load(file, mono=True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        file_length = mfcc.shape[-1]
        no_segments = math.ceil(file_length / step)
        for i in range(no_segments):
            data = np.array([])
            l = i*step
            r = min(i*step + sample_length, file_length)
            # print(l)
            # print(r)
            if r - l == sample_length:
                data = mfcc[:,l:r]
                data = np.append(data, spectral_center[:,l:r], axis=0)
                data = np.append(data, chroma[:, l:r], axis=0)
                data = np.append(data, spectral_contrast[:, l:r], axis=0)

                data = np.append(data, chroma_cqt[:, l:r], axis=0)
                data = np.append(data, chroma_cens[:, l:r], axis=0)
                data = np.append(data, melspectrogram[:, l:r], axis=0)
                data = np.append(data, rmse[:, l:r], axis=0)
                data = np.append(data, spec_bw[:, l:r], axis=0)
                data = np.append(data, rolloff[:, l:r], axis=0)
                data = np.append(data, flatness[:, l:r], axis=0)
                data = np.append(data, poly_features[:, l:r], axis=0)
                data = np.append(data, tonnetz[:, l:r], axis=0)
                data = np.append(data, zcr[:, l:r], axis=0)

                # print('aasda')
            elif r - l > sample_length / 4:
                
                mfcc_pad = np.zeros((mfcc.shape[0], sample_length))
                mfcc_pad[:, 0:(r-l)] = mfcc[:, l:r]
                data = mfcc_pad
                
                spectral_center_pad = np.zeros((spectral_center.shape[0], sample_length))
                spectral_center_pad[:, 0:(r-l)] = spectral_center[:, l:r]
                data = np.append(data, spectral_center_pad, axis=0)
                
                chroma_pad = np.zeros((chroma.shape[0], sample_length))
                chroma_pad[:, 0:(r-l)] = chroma[:, l:r]
                data = np.append(data, chroma_pad, axis=0)
                
                spectral_contrast_pad = np.zeros((spectral_contrast.shape[0], sample_length))
                spectral_contrast_pad[:, 0:(r-l)] = spectral_contrast[:, l:r]
                data = np.append(data, spectral_contrast_pad, axis=0)

                chroma_cqt_pad = np.zeros((chroma_cqt.shape[0], sample_length))
                chroma_cqt_pad[:, 0:(r-l)] = chroma_cqt[:, l:r]
                data = np.append(data, chroma_cqt_pad, axis=0)
                
                chroma_cens_pad = np.zeros((chroma_cens.shape[0], sample_length))
                chroma_cens_pad[:, 0:(r-l)] = chroma_cens[:, l:r]
                data = np.append(data, chroma_cens_pad, axis=0)

                melspectrogram_pad = np.zeros((melspectrogram.shape[0], sample_length))
                melspectrogram_pad[:, 0:(r-l)] = melspectrogram[:, l:r]
                data = np.append(data, melspectrogram_pad, axis=0)

                rmse_pad = np.zeros((rmse.shape[0], sample_length))
                rmse_pad[:, 0:(r-l)] = rmse[:, l:r]
                data = np.append(data, rmse_pad, axis=0)

                spec_bw_pad = np.zeros((spec_bw.shape[0], sample_length))
                spec_bw_pad[:, 0:(r-l)] = spec_bw[:, l:r]
                data = np.append(data, spec_bw_pad, axis=0)

                rolloff_pad = np.zeros((rolloff.shape[0], sample_length))
                rolloff_pad[:, 0:(r-l)] = rolloff[:, l:r]
                data = np.append(data, rolloff_pad, axis=0)

                flatness_pad = np.zeros((flatness.shape[0], sample_length))
                flatness_pad[:, 0:(r-l)] = flatness[:, l:r]
                data = np.append(data, flatness_pad, axis=0)

                poly_features_pad = np.zeros((poly_features.shape[0], sample_length))
                poly_features_pad[:, 0:(r-l)] = poly_features[:, l:r]
                data = np.append(data, poly_features_pad, axis=0)

                tonnetz_pad = np.zeros((tonnetz.shape[0], sample_length))
                tonnetz_pad[:, 0:(r-l)] = tonnetz[:, l:r]
                data = np.append(data, tonnetz_pad, axis=0)

                zcr_pad = np.zeros((zcr.shape[0], sample_length))
                zcr_pad[:, 0:(r-l)] = zcr[:, l:r]
                data = np.append(data, zcr_pad, axis=0)
                # print('123')
                
            # print(f'data shape {data.shape}')
            if data.shape[0] != 0:
                data = np.array([data.T])
                if return_data_x.shape[0] == 0:
                    return_data_x = data
                else:
                    return_data_x = np.append(return_data_x, data, axis=0)
                return_data_y = np.append(return_data_y, genre)
            # print(f're data {return_data_x.shape}')
        return return_data_x, return_data_y
        
    def extract_features_files(self, files, save_file):
        sample_length = 2000
        step = int(sample_length / 2)
        X_train_data = np.array([])
        seg_data = np.array([])
        count = 0
        for i,prog_file in enumerate(files):
            file_data_x, _ = self.extract_feature(prog_file, 0, sample_length, step)
            no_seg = file_data_x.shape[0]
            print("Extracted features audio track %i of %i." % (i + 1, len(files)))
            if X_train_data.shape[0] == 0:
                X_train_data = file_data_x
            else:
                X_train_data = np.append(X_train_data, file_data_x, axis=0)
            
            if seg_data.shape[0] == 0:
                seg_data = np.append(seg_data, no_seg)
            else:
                seg_data = np.append(seg_data, seg_data[-1] + no_seg)

            if X_train_data.shape[0] >= 300:
                with open(f'{save_file}_{count}.npy', 'wb') as f:
                    np.save(f, X_train_data)
                X_train_data = np.array([])
                count = count + 1

            print(f'x {X_train_data.shape}')

        if X_train_data.shape[0] > 0:
            with open(f'{save_file}_{count}.npy', 'wb') as f:
                np.save(f, X_train_data)
            X_train_data = np.array([])
        return X_train_data, seg_data

    
    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            directory = "%s\\%s" % (dir_folder, file)
            list_of_audio.append(directory)
        return list_of_audio

if len(sys.argv) > 1:
    path_id = int(sys.argv[1])

    # normalize array or not
    normalize = False
    if len(sys.argv) > 2:
        if int(sys.argv[2]) == 1:
            normalize = True
        else:
            normalize = False
    
    # load the whole song or only 10 minutes
    load_full = False
    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            load_full = True
        else:
            load_full = False

    genre_features = GenreFeatureDataVal(normalize, load_full)
    genre_features.load_and_save_data(path_id)

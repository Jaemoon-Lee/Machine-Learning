Here is description of each files

- path.txt : this file contains paths to prog and nonprog folder, first line is prog path and second line is nonprog path. Before running code, we have to setup the paths in this file.

- extract_lstm.py : this file will extract features for lstm model from files in prog path and nonprog path in path.txt

- extract_nn.py : this file will extract feature for neural network model and random forest model from files in prog path and nonprog path in path.txt

After correctly setting paths in path.txt, run extract_lstm.py and extract_nn.py without any other argument, e.g: "py extract_lstm.py" and "py extract_nn.py" to extract features. The code will create folder "extract" and store features in there. With extract_nn, the code will generate a  data_full_features4.csv file, which is understandable. With extract_lstm, the code will generate a sequence of files in the format "dataTest_X_Y.npy" and a file "dataTest_seg_X.npy" for each path in path.txt, where X is index of the path in path_id (starting from 0). 

To train the model, the codes are:

- lstm.py : change the paths to "dataTest_X_Y.npy" file in the beginning of the code for both prog and nonprog. Then simply run "py lstm.py". The code will store model after each epoch in folder "model/ZZZ" where ZZZ is milliseconds measurement of the time we start training.
- nn_model.py : change the paths to *.csv files that were generated in the beginning of the code, then simply run "py nn_model.py". The code will store the best model in term of test set evaluation in folder "model/ZZZ" where ZZZ is milliseconds measurement of the time we start training.

To evaluate the model, the codes are:

- evaluate_lstm.py : changes the paths in the beginning of the code to "dataTest_X_Y.npy" of test set and "***.hdf5" of the model we would like to evaluate. Then simply run "py evaluate_lstm.py"

- evaluate_nn.py : changes the paths in the beginning of the code to *.csv of test set and "***.hdf5" of the model we would like to evaluate. Then simply run "py evaluate_nn.py"
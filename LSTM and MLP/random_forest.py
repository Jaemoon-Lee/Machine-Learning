import itertools
import numpy, scipy, matplotlib.pyplot as plt, pandas, librosa,sklearn

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

data_train=pandas.read_csv('data_full_data_full_features_3.csv')
# data_train.head()
GENRES=['prog', 'nonprog']
data_train = data_train.iloc[:,:]
number_of_rows,number_of_cols = data_train.shape


data_validation=pandas.read_csv('./extract/data_full_features.csv')
# data_validation.head()
GENRES=['prog', 'nonprog']
data_validation = data_validation.iloc[:,:]
number_of_rows_validation,number_of_cols_validation = data_validation.shape


data_train_values=numpy.array(data_train)
data_validation_values=numpy.array(data_validation)

train_x=data_train.iloc[:,:number_of_cols-1]
train_y=data_train.iloc[:,number_of_cols-1]

validation_x=data_validation.iloc[:,:number_of_cols_validation-1]
validation_y=data_validation.iloc[:,number_of_cols_validation-1]

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.1,
    max_depth=2, random_state=0).fit(train_x, train_y)
training_score = clf.score(train_x,train_y)
validation_score = clf.score(validation_x,validation_y)
print("Training Score :",training_score)
print("Test Score :",validation_score)
validationpredict = clf.predict(validation_x)
print(validationpredict)
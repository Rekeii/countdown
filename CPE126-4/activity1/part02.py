import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Loading dataset from csv file.
data = pd.read_csv("student-mat.csv", sep=";")

# Various features selected from dataset.
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# What we want to predict. 
predict = "G3"

# Creation of feature and label arrays.
# We drop G3 from features since it's what we want to predict. 
# The 1 after it indicates that we are dropping a column, not a row.
# Y is the target label array, the values we want to predict, G3.
X = np.array(data.drop([predict], axis=1)) # edited 1 -> axis=1 
y = np.array(data[predict])


# Splitting data into training and testing sets.
# Scikit, train_test_split will randomly split the data.
# 90/10 split, 90 for training, 10 for testing.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# This file only sets up the model and splits the dataset.
# The actual training and testing of the model is done in part04.py
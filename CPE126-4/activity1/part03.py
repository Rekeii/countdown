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

# Creation of linear regression model.
linear = linear_model.LinearRegression()

# Training the model.
# `fit()` computes the best fit line for the data. 
# (coefficients and intercept to minimize residual sum of squares)
# ordinary least squares.
linear.fit(x_train, y_train)

# evaluating the model; coefficient of determination
# 0-1, 1 is perfect prediction.
acc = linear.score(x_test, y_test)
print('Accuracy:', acc)

# learned coefficient, weight for each feature.
# intercept, line crosses the y-axis.
print('Coefficient:', linear.coef_)
print('Intercept:', linear.intercept_)

# generate predictions based on the test features using the learned model.
predictions = linear.predict(x_test)

# compare predictions to actual values.
# loops through each test example.
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    
# example output: 17.503763014466976 [16 17  1  0  4] 18
# meaning: predicted 17.5, student had 16 in G1, 17 in G2, 1 absence, 0 failures, studytime of 4
# but actually got 18 in G3 (final grade).
# So, its 17.5 (Predicted by Model) vs 18 (Actual), pretty close.

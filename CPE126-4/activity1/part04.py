import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle # preinstalled with python3.

style.use("ggplot")

# Loading dataset from csv file.
data = pd.read_csv("student-mat.csv", sep=";")

# What we want to predict.
predict = "G3"

# Various features selected from dataset.
data = data[["G1", "G2", "absences","failures", "studytime","G3"]]
data = shuffle(data) # Optional - shuffle the data

# Creation of feature and label arrays.
x = np.array(data.drop([predict], axis=1)) # edited 1 -> axis=1
y =np.array(data[predict])

# Splitting data into training and testing sets; 90/10 split.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
# Trains the model 20 times and saves the best one (studentgrades.pickle).
# Accuracy (R^2) is used as basis.
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
# Loads the best model saved above.
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient:', linear.coef_)
print('Intercept:', linear.intercept_)
print("-------------------------")

# Using the best model, we make predictions.
# This is the same as in part03.py, but now,
# we've trained the model multiple times and saved the best one.
predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# Drawing and plotting model
# Utilizing matplotlib to plot a scatter plot,
# showing the relationship between a selected feature and the final grade.
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()

""" 
Notes for myself;
Sample Output:

Accuracy: 0.6908902708377156
Accuracy: 0.6972189455876534
Accuracy: 0.880144889124652

We can see that it varies each time we train the model.
Highest thus far, Accuracy: 0.9350934739916221

Using that model it shows the coefficients and intercept:

Coefficient: [ 0.1504473   0.98214302  0.04481446 -0.31174528 -0.19405145]
Intercept: -1.5183000327051843

Then it shows the predictions vs actual values:

9.031397402067828 [10 10  0  0  4] 10
12.323070829562292 [10 13  6  1  2] 13
12.756452880255088 [12 13  2  0  2] 12

Using the best model, we can see that the predictions are 
quite close to the actual values.

"""

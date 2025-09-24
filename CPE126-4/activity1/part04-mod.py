# Added improvements:
# - Trains the model multiple times and saves the best one. (20->50)
# - Displays best accuracy and average accuracy.
# - Formatted output for predictions.
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle # preinstalled with python3.
import random

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
scores = []
best = 0 # added to display best accuracy.
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1
        )

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    scores.append(acc) # added to calculate average accuracy.
    # print("Accuracy: " + str(acc)) removed to reduce clutter.

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
# Loads the best model saved above.
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print("Best Accuracy: ", best) # added
print("Average Accuracy:", sum(scores)/len(scores))
print('Coefficient:', linear.coef_)
print('Intercept:', linear.intercept_)
print("-------------------------")

# Using the best model, we make predictions.
# This is the same as in part03.py, but now,
# we've trained the model multiple times and saved the best one.
predicted= linear.predict(x_test)

# added formatted output
print("-" * 65)
print(f"{'Predicted':>10} {'G1':>6} {'G2':>6} {'Absences':>9} {'Failures':>9} {'Studytime':>10} {'Actual':>7}")
print("-" * 65)
for x in range(len(predicted)):
    print(f"{predicted[x]:>10.2f} {x_test[x][0]:>6.2f} {x_test[x][1]:>6.2f} {x_test[x][2]:>9.2f} {x_test[x][3]:>9.2f} {x_test[x][4]:>10.2f} {y_test[x]:>7}")


# Drawing and plotting model
# Utilizing matplotlib to plot a scatter plot,
# showing the relationship between a selected feature and the final grade.
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()




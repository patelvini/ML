from sklearn.linear_model import LinearRegression
import os
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


X = np.array([1,2,3,4,5])
Y = np.array([3,4,2,4,5])

# Cannot use Rank 1 matrix in scikit learn
# Reshape the input data into 2D array

X = X.reshape((len(X), 1))
Y = Y.reshape((len(Y), 1))

# Creating Model
reg = LinearRegression()

# Fitting training data
reg = reg.fit(X, Y)

# print the coefficients
print("Slope : ",reg.coef_)
print("Y-intercept : " ,reg.intercept_)

# Y Prediction
Y_pred = list(reg.predict(X))

print("Pridictions ", end = ": ")

for i in range(len(Y_pred)):
	print(round(float(Y_pred[i]),3), end= ", ")

accuracy  = reg.score(Y,Y_pred)

print("\n",accuracy * 100, "%")

exit()
 
# Calculating R2 Score
print("\nR-Squared value : ",r2_score(Y, Y_pred))

plt.plot(X,Y_pred,color='red',label='Linear Regression')
plt.scatter(X,Y,c='b',label='Scatter Plot')
plt.xlabel("Head Size")
plt.ylabel("Brain Weight")
plt.legend()
plt.show() 
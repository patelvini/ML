import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegressor:

	def calculateSlopeAndIntercept(self, X, Y):
		# mean_x = round(np.mean(X),2)
		# mean_y  = round(np.mean(Y),2)
		mean_x = 0
		mean_y  = 0

		for i in range(0,len(X)):
			mean_x=mean_x+X[i]
			mean_y=mean_y+Y[i]

		mean_x=round(mean_x/len(X),2)
		mean_y=round(mean_y/len(Y),2)

		n = len(X)

		# slope (m) = sum(x[i] - mean_x)(y[i] - mean_y)/sum(x[i]-mean_x ** 2)

		numerator = 0
		denominator = 0

		for i in range(n):
			numerator += ((X[i] - mean_x) * (Y[i] - mean_y))
			denominator += ((X[i] - mean_x) ** 2)

		m = round((numerator/denominator),2)

		c = mean_y - (m * mean_x)

		return m,c

	def predict(self, X, m, c):
		pred_y = []

		for i in range(len(X)):
			pred_y.append(c + (m * X[i]))

		return(pred_y)

	def calculate_R_square(self, Y_test, Y_pred):

		mean_y = np.mean(Y_test)

		# R-square = 1 - [(sum(y_test[i] - y_pred[i] )**2)/(sum(y_test[i] - mean_y)**2)

		numerator = 0
		denominator = 0

		for i in range(len(Y_test)):
			numerator += ((Y_test[i] - Y_pred[i]) ** 2)
			denominator += ((Y_test[i] - mean_y) ** 2)

		r2 = 1 - (numerator/denominator)

		return r2

	def RMSE(self, Y_test, Y_pred):
		rmse_sum = 0
		for i in range(len(Y_test)):
			rmse_sum += ((Y_test[i] - Y_pred[i])**2)
		rmse = (rmse_sum / len(Y_test)) ** (1/2)
		return rmse


if __name__ == '__main__':
	
	X = [1,2,3,4,5]
	Y = [3,4,2,4,5]

	linear_regressor = LinearRegressor()

	m,c = linear_regressor.calculateSlopeAndIntercept(X, Y)

	print("Slope of regression line : ", m)
	print("Y-intercept of the line : ", c)

	predictions = linear_regressor.predict(X, m, c)

	print("Predictions : ",predictions)

	r2 =linear_regressor.calculate_R_square(Y, predictions)

	print("\nR-Squared value : ",r2)

	plt.plot(X,predictions,color='red',label='Linear Regression')
	plt.scatter(X,Y,c='b',label='Scatter Plot')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.legend()
	plt.show()
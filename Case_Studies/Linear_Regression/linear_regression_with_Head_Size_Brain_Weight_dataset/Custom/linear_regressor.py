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

	def train_test_split(self, dataset):
		training_data = dataset.iloc[:191].reset_index(drop = True)
		testing_data = dataset.iloc[191:].reset_index(drop = True)

		return training_data, testing_data

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
		

if __name__ == '__main__':
	
	path = os.getcwd()
	for i in range(3):
		path = os.path.dirname(path)
	data = pd.read_csv( path + '/Datasets/HeadBrain.csv')

	# print(data)

	# print(data.shape)

	# print(data.info())

	# plt.figure(figsize = (10,10))
	# sns.scatterplot(y='Brain Weight(grams)', x='Head Size(cm^3)', data = data)
	# plt.show()

	# print(X)
	# print(Y)

	linear_regressor = LinearRegressor()

	training_data = linear_regressor.train_test_split(data)[0]
	testing_data = linear_regressor.train_test_split(data)[1]

	X_train = training_data['Head Size(cm^3)']
	Y_train = training_data['Brain Weight(grams)']

	X_test = testing_data['Head Size(cm^3)']
	Y_test = testing_data['Brain Weight(grams)']

	m,c = linear_regressor.calculateSlopeAndIntercept(X_train, Y_train)

	predictions = linear_regressor.predict(X_test, m, c)

	print(predictions)

	r2 =linear_regressor.calculate_R_square(Y_test, predictions)

	print("\nR-Squared value : ",r2)
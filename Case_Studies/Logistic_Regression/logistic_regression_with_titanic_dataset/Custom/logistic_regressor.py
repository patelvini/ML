import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_tnc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


class LogisticRegression:

	def sigmoid(self, x):
		return 1 / (1+np.exp(-x))

	def net_input(self, theta, x):
		return np.dot(x, theta)

	def probability(self, theta, x):
		return self.sigmoid(self.net_input(theta, x))

	def cost_function(self, theta, x, y):
		m = x.shape[0]
		total_cost = -(1/m) * np.sum(y * np.log(self.probability(theta, x)) + (1-y)*np.log(1-self.probability(theta, x)))

		return total_cost

	def gradient(self, theta, x, y):
		m = x.shape[0]
		return (1/m)* np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

	def fit(self, x, y, theta):
		opt_wights = fmin_tnc(func=self.cost_function, x0 = theta, fprime=self.gradient, args=(x, y.flatten()))
		return opt_wights[0]

	def predict(self, x):
		theta = parameters[:,np.newaxis]
		return self.probability(theta, x)

	def evaluation_metrics(self, cnf_matrix):
		TN = cnf_matrix[0][0]
		FP = cnf_matrix[0][1]
		FN = cnf_matrix[1][0]
		TP = cnf_matrix[1][1]

		accuracy = (TP + TN)/(TP + TN + FP + FN)
		precision = TP/(TP + FP)
		recall = TP/(TP + FN)
		return accuracy, precision, recall


if __name__ == '__main__':

	# load data
	path = os.getcwd()
	for i in range(3):
		path = os.path.dirname(path)
	data = pd.read_excel( path + '/Datasets/Titanic_Dataset.xlsx')

	# X = feature values, all the columns except the last column
	X = data.iloc[:,:-1] 

	# y = target column, Survived for this example
	y = data.iloc[:,-1]

	# prepare data for training

	X = np.c_[np.ones((X.shape[0], 1)), X]
	y = y[:, np.newaxis]
	theta = np.zeros((X.shape[1], 1))

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

	log_reg = LogisticRegression()

	threhold = 0.5
	parameters = log_reg.fit(X_train, y_train, theta)

	print("\n\nparameters : ",parameters)

	predicted_classes = (log_reg.predict(X_test) >= threhold).astype(int)

	clf_report = classification_report(y_test, predicted_classes)

	print("\nclassification_report : \n\n",clf_report)

	cnf_matrix = confusion_matrix(y_test,predicted_classes)

	print("\nConfusion Matrix : \n\n",cnf_matrix)


	accuracy, precision, recall = log_reg.evaluation_metrics(cnf_matrix)

	print("\naccuracy : ",round( accuracy * 100 , 3), "%")
	print("\nprecision : ",round( precision * 100, 3), "%")
	print("\nrecall : ",round( recall * 100, 3), "%")





	


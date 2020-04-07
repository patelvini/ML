import numpy as np 
import pandas as pd 
import os


class LogisticRegression:

	def __init__(self, lr = 0.01, n_iter= 100):
		self.lr = lr
		self.n_iter = n_iter


	def train_test_split(self,dataset):

		training_data = dataset.iloc[:1200].reset_index(drop = True)
		testing_data = dataset.iloc[1200:].reset_index(drop = True)

		X_train = training_data[['Passengerid','Age','Fare','Sex','sibsp','zero','Pclass','Embarked']]
		y_train = training_data['Survived']
		
		X_test = testing_data[['Passengerid','Age','Fare','Sex','sibsp','zero','Pclass','Embarked']]
		y_test = testing_data['Survived']

		return X_train.values, y_train.values, X_test.values, y_test.values

	def sigmoid(self, X):
		y = 1/(1 + np.exp(-X))
		return y

	def linear(self, X):
		X = np.dot(X, self.weights) + self.bias
		return X

	def initialize_weights(self, X):
		self.weights = np.random.rand(X.shape[1],1)
		self.bias = np.zeros((1,))

	def normalize(self, X):
		X = (X - self.x_mean) / self.x_stddev
		return X

	def fit(self, X_train, y_train):

		self.initialize_weights(X_train)

		self.x_mean = X_train.mean(axis=0).T

		self.x_stddev = X_train.std(axis=0).T
        
        # normalize data

		X_train = self.normalize(X_train)

        # Run gradient descent for n iterations
		for i in range(self.n_iter):
            # make normalized predictions
			probs = self.sigmoid(self.linear(X_train))

			print(probs)
			diff = probs - y_train

			print(diff)

            # d/dw and d/db of mse
			delta_w = np.mean(diff * X_train, axis=0, keepdims=True).T
			delta_b = np.mean(diff)

            # update weights
			self.weights = self.weights - self.lr * delta_w
			self.bias = self.bias - self.lr * delta_b
		return self


	def predict(self, X):
		y_pred = self.sigmoid(X)
		if y_pred >= 0.5:
			return 1
		else:
			return 0

	def accuracy(self, y_test, y_pred):
		return np.mean(y_test == y_pred)


if __name__ == '__main__':

	path = os.getcwd()
	
	for i in range(3):
		path = os.path.dirname(path)
	
	data = pd.read_excel( path + '/Datasets/Titanic_Dataset.xlsx')

	data.sort_values(by = 'Age', inplace = True)

	# print(data)

	log_regressor = LogisticRegression()

	X_train, y_train, X_test, y_test = log_regressor.train_test_split(data)

	log_regressor.fit(X_train, y_train)

	y_pred = log_regressor.predict(X_test)

	accuracy = log_regressor.accuracy(y_test, y_pred)

	print("accuracy : ", accuracy*100,"%")




import pandas as pd
import numpy as np
import pickle
import os
import operator
from sklearn.datasets import load_iris
import random


class KNNClassifier:

	def train_test_split(self,dataset):

		training_data = dataset.iloc[:70].reset_index(drop = True)

		testing_data = dataset.iloc[70:].reset_index(drop = True)

		return training_data, testing_data

	
	def euclidian_distance(self, data1, data2, length):

		distance = 0

		for i in range(length):

			distance += ((data1[i] - data2[i])**2)
		distance = (distance ** (1/2))

		return distance

	def getKNeighbors(self, trainingSet, testInstance, k):

		distances = []

		length = len(testInstance)

		for i in range(len(trainingSet)):

			dist = self.euclidian_distance(testInstance, trainingSet[i], length)

			distances.append((trainingSet[i], dist))

		distances.sort(key = operator.itemgetter(1))

		neighbors = []

		for i in range(k):
			neighbors.append(distances[i][0])

		return neighbors


	def fit(self, train_data):

		trainingSet = []

		for index, rows in train_data.iterrows():
			my_list = [rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width, rows.species]

			trainingSet.append(my_list)

		return trainingSet

	def predict(self, neighbors):

		classVotes = {}

		for i in range(len(neighbors)):

			response = neighbors[i][-1]

			if response in classVotes:
				classVotes[response] +=1
			else:
				classVotes[response] = 1

		sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse = True)

		return sortedVotes[0][0]

	def getAccuracy(self, testSet, predictions):

		correct = 0

		for i in range(len(testSet)):
			if testSet[i] is predictions[i]:
				correct += 1

		accuracy = correct/float(len(testSet)) * 100.0

		return accuracy

if __name__ == '__main__':
	
	path = os.getcwd()

	for i in range(3):
		path = os.path.dirname(path)

	data = pd.read_csv( path + '/Datasets/IRIS.csv')

	data.sort_values(by = 'sepal_width', inplace = True)

	KNN = KNNClassifier()

	training_data =  KNN.train_test_split(data)[0]
	testing_data = KNN.train_test_split(data)[1]

	test_classes = []
	test_data = []

	for index, rows in testing_data.iterrows():
		my_list = [rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]
		test_classes.append(rows.species)
		test_data.append(my_list)

	trainSet = KNN.fit(training_data)

	k = int(input(" Enter value for the k : "))

	predictions = []

	for i in range(len(test_data)):
		neighbors = KNN.getKNeighbors(trainSet, test_data[i], k)

		predictions.append(KNN.predict(neighbors))

		print(neighbors)
		print(predictions)
		print("*"*50)


	accuracy = KNN.getAccuracy(test_classes, predictions)

	print("\n Accuracy = ", accuracy, "%")

	testInstance = [5.1,3.3,1.4,0.21]

	neighbors = KNN.getKNeighbors(trainSet, testInstance, k)

	print(" Nearest Neighbors = ")

	for neighbor in neighbors:
		print(neighbor)


	prediction = KNN.predict(neighbors)

	print("\n Prediction for test input : ",prediction)

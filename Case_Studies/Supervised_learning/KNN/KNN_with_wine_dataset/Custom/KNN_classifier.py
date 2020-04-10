import pandas as pd
import numpy as np
import pickle
import os
import operator
import random
import Commands as cmd
import math
from scipy.spatial import distance
from sklearn.utils import shuffle

class KNNClassifier:

	def train_test_split(self,dataset):
		training_data = dataset.iloc[:106].reset_index(drop = True)
		testing_data = dataset.iloc[106:].reset_index(drop = True)
		trainingSet = []
		test_classes = []
		test_data = []

		for index, rows in training_data.iterrows():
			my_list = [rows.Alcohol, rows.Malic_acid, rows.Ash, rows.Alcalinity_of_ash, rows.Magnesium, rows.Total_phenols, rows.Flavanoids, rows.Nonflavanoid_phenols, rows.Proanthocyanins, rows.Color_intensity, rows.Hue, rows.OD280_OD315_of_diluted_wines, rows.Proline, rows.Class]
			trainingSet.append(my_list)

		for index, rows in testing_data.iterrows():
			my_list = [rows.Alcohol, rows.Malic_acid, rows.Ash, rows.Alcalinity_of_ash, rows.Magnesium, rows.Total_phenols, rows.Flavanoids, rows.Nonflavanoid_phenols, rows.Proanthocyanins, rows.Color_intensity, rows.Hue, rows.OD280_OD315_of_diluted_wines, rows.Proline]
			test_classes.append(rows.Class)
			test_data.append(my_list)
		return trainingSet,test_data,test_classes

	
	# def euclidean_distance(self, data1, data2, length):

	# 	data2 = data2[:-1]
	# 	distance1 = float(distance.euclidean(data1, data2))

	# 	return distance1

	# def euclidean_distance(self, data1, data2, length):

	# 	distance1 = 0
	# 	for i in range(length):
	# 		distance1 += ((data1[i] - data2[i])**2)
	# 	distance1 = (distance1 ** (1/2))

	# 	return distance1

	# def manhattan(self, data1, data2, length):
	# 	data2 = data2[:-1]
	# 	distance1 = float(distance.cityblock(data1 , data2))

	# 	return distance1

	def manhattan(self, data1, data2, length):

		distance1 = 0
		for i in range(length):
			distance1 += abs(data1[i]-data2[i])
		return distance1

	def getKNeighbors(self, trainingSet, testInstance, k):
		distances = []
		length = len(testInstance)
		for i in range(len(trainingSet)):
			dist = self.manhattan(testInstance, trainingSet[i], length)
			distances.append((trainingSet[i], dist))
		distances.sort(key = operator.itemgetter(1))
		neighbors = []
		for i in range(k):
			neighbors.append(distances[i][0])
		return neighbors

	def fit(self, train_data):

		trainingSet = train_data
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
			if testSet[i] == predictions[i]:
				correct += 1
		accuracy = correct/float(len(testSet)) * 100.0
		return accuracy

if __name__ == '__main__':
	
	path = os.getcwd()
	for i in range(3):
		path = os.path.dirname(path)
	data = pd.read_csv( path + '/Datasets/WinePredictor.csv', names = ['Class','Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280_OD315_of_diluted_wines','Proline'])

	data = shuffle(data, random_state = 1)

	KNN = KNNClassifier()

	training_data, testing_data, test_classes =  KNN.train_test_split(data)


	k = 5
	
	if cmd.args.train:
		trainSet = KNN.fit(training_data)
		print("Model trained Successfully !!!")
	
	elif cmd.args.test:
		print("test_classes :     ",test_classes)
		predictions = []
		trainSet = KNN.fit(training_data)
		for i in range(len(testing_data)):
			neighbors = KNN.getKNeighbors(trainSet, testing_data[i], k)
			predictions.append(KNN.predict(neighbors))

		print("predicted_classes :",predictions)
		accuracy = KNN.getAccuracy(test_classes, predictions)
		print("\nAccuracy = ", accuracy, "%")
	
	elif cmd.args.predict:
		trainSet = KNN.fit(training_data)
		testInstance = [13.82,1.56,2.52,21,114,2.85,3.4,0.3,1.71,6.6,1.12,2.53,1135]
		neighbors = KNN.getKNeighbors(trainSet, testInstance, k)
		print("Nearest Neighbors = ")
		for neighbor in neighbors:
			print(neighbor)
		prediction = KNN.predict(neighbors)
		print("\nPrediction for test input : ",prediction)
	
	else:
		print("No argument given !!!")
		print(" --train for training.")
		print(" --test for check the accuracy.")
		print(" --predict for predict label for given input")
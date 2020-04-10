import pandas as pd
import numpy as np
import pickle
import os
import operator
from sklearn.datasets import load_iris
import random
import Commands as cmd
import math
from scipy.spatial import distance

class KNNClassifier:

	def train_test_split(self,dataset):
		training_data = dataset.iloc[:31].reset_index(drop = True)
		testing_data = dataset.iloc[31:].reset_index(drop = True)
		trainingSet = []
		test_classes = []
		test_data = []
		for index, rows in training_data.iterrows():
			my_list = [rows.Whether, rows.Temperature, rows.Play]
			trainingSet.append(my_list)

		for index, rows in testing_data.iterrows():
			my_list = [rows.Whether, rows.Temperature]
			test_classes.append(rows.Play)
			test_data.append(my_list)
		return trainingSet,test_data,test_classes

	
	# def euclidean_distance(self, data1, data2, length):

	# 	data2 = data2[:-1]
	# 	distance1 = float(distance.euclidean(data1, data2))

	# 	return distance1

	def euclidean_distance(self, data1, data2, length):

		distance1 = 0
		for i in range(length):
			distance1 += ((data1[i] - data2[i])**2)
		distance1 = (distance1 ** (1/2))

		return distance1

	# def manhattan(self, data1, data2, length):
	# 	data2 = data2[:-1]
	# 	distance1 = float(distance.cityblock(data1 , data2))

	# 	return distance1

	# def manhattan(self, data1, data2, length):

	# 	distance1 = 0
	# 	for i in range(length):
	# 		distance1 += abs(data1[i]-data2[i])
	# 	return distance1

	def getKNeighbors(self, trainingSet, testInstance, k):
		distances = []
		length = len(testInstance)
		for i in range(len(trainingSet)):
			dist = self.euclidean_distance(testInstance, trainingSet[i], length)
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
			if testSet[i] is predictions[i]:
				correct += 1
		accuracy = correct/float(len(testSet)) * 100.0
		return accuracy

if __name__ == '__main__':
	
	path = os.getcwd()
	for i in range(3):
		path = os.path.dirname(path)
	data = pd.read_csv( path + '/Datasets/Play_Predictor.csv')

	data = data.replace(to_replace = "Sunny", value = 1)
	data = data.replace(to_replace = "Overcast", value = 2)
	data = data.replace(to_replace = "Rainy", value = 3)

	data = data.replace(to_replace = "Hot", value = 1)
	data = data.replace(to_replace = "Mild", value = 2)
	data = data.replace(to_replace = "Cool", value = 3)
	

	KNN = KNNClassifier()
	training_data, testing_data, test_classes =  KNN.train_test_split(data)
	k = 5

	if cmd.args.train:
		trainSet = KNN.fit(training_data)
		print("Model trained Successfully !!!")
	elif cmd.args.test:
		predictions = []
		trainSet = KNN.fit(training_data)
		for i in range(len(testing_data)):
			neighbors = KNN.getKNeighbors(trainSet, testing_data[i], k)
			predictions.append(KNN.predict(neighbors))
		accuracy = KNN.getAccuracy(test_classes, predictions)
		print("\nAccuracy = ", accuracy, "%")
	elif cmd.args.predict:
		print("for Whether,\n\t1 -> Sunny \n\t2 -> Overcast \n\t3 -> Rainy" )
		print("for Temperature,\n\t1 -> Hot \n\t2 -> Mild \n\t3 -> Cool")
		trainSet = KNN.fit(training_data)
		testInstance = [cmd.args.whether,cmd.args.temperature]
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
		print(" -w for input Whether.")
		print(" -t for input Temperature.")
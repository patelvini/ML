import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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

all_inputs = data[['Whether','Temperature']]
target_lables = data['Play']

# you can choose the number k as you want

# if we are not providing the value for n_neighbors then bydefault the value for k is 5.

KNN = KNeighborsClassifier(n_neighbors = 5)


(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, target_lables, train_size=0.7, random_state=0)

KNN.fit(train_inputs,train_classes)

predictions = KNN.predict(test_inputs)

print("predictions : ",predictions)
print("test_classes ", end=": ")

for i in test_classes:
	print(i,end=", ")

accuracy = round(KNN.score(test_inputs,test_classes),4) * 100

print("\nAccuracy = ", accuracy, "%")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd
from sklearn.utils import shuffle

path = os.getcwd()
for i in range(3):
	path = os.path.dirname(path)
data = pd.read_csv( path + '/Datasets/WinePredictor.csv', names = ['Class','Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280_OD315_of_diluted_wines','Proline'])

data = shuffle(data, random_state = 0)

all_inputs = data[['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280_OD315_of_diluted_wines','Proline']]

target_labels = data['Class']

# you can choose the number k as you want

# if we are not providing the value for n_neighbors then bydefault the value for k is 5.

KNN = KNeighborsClassifier(n_neighbors = 5)

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, target_labels, train_size=0.6, random_state=1)


print("test_classes :",list(test_classes))

KNN.fit(train_inputs,train_classes)

predictions = KNN.predict(test_inputs)

print("Predictions :",list(predictions))

accuracy = round(KNN.score(test_inputs,test_classes),4) * 100

print("Accuracy = ", accuracy, "%")

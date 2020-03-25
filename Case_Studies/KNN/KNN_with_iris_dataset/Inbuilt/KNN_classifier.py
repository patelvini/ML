from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

all_inputs = iris.data

target_lables = iris.target

KNN = KNeighborsClassifier(n_neighbors = 3)


(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, target_lables, train_size=0.7, random_state=1)

print(train_classes)

print(test_classes)

KNN.fit(train_inputs,train_classes)

KNN.predict(test_inputs)

accuracy = round(KNN.score(test_inputs,test_classes),4) * 100

print("Accuracy = ", accuracy, "%")

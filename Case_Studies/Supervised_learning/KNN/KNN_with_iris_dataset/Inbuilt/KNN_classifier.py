from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

all_inputs = iris.data

target_lables = iris.target

# you can choose the number k as you want

# if we are not providing the value for n_neighbors then bydefault the value for k is 5.

KNN = KNeighborsClassifier(n_neighbors = 3)


(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, target_lables, train_size=0.7, random_state=1)

print('*'*50,"Train classes",'*'*50)
print(train_classes)

print('*'*50,"Test classes",'*'*50)
print(test_classes)

KNN.fit(train_inputs,train_classes)

KNN.predict(test_inputs)

accuracy = round(KNN.score(test_inputs,test_classes),4) * 100

print("Accuracy = ", accuracy, "%")

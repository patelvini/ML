from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = [[35,1], [47,1], [90,0], [48,1], [90,0], [35,1], [92,0], [35,1], [35,1], [35,1], [96,0], [43,1], [110,0], [35,1], [95,0]]

target_label = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]


# you can choose the number k as you want

# if we are not providing the value for n_neighbors then bydefault the value for k is 5.

KNN = KNeighborsClassifier(n_neighbors = 3)


(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(data, target_label, train_size=0.7, random_state=1)

print("Train classes : ")
print(train_classes)

print("Test classes")
print(test_classes)

KNN.fit(train_inputs,train_classes)

KNN.predict(test_inputs)

accuracy = round(KNN.score(test_inputs,test_classes),4) * 100

print("Accuracy = ", accuracy, "%")

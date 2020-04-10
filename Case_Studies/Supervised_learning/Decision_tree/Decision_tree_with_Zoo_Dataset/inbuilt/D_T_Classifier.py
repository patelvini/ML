from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd


# True : 1
# False : 0



data = pd.read_csv('zoo.csv',names = ['animal_name','hair','feathers','eggs','milk','airbone','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','class',])

data=data.drop('animal_name',axis=1)

features = data[['hair','feathers','eggs','milk','airbone','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]
target = data['class']


train_inputs, test_inputs , train_classes, test_classes = train_test_split(features, target, train_size = 0.8, random_state = 1)

model = DecisionTreeClassifier()

model.fit(train_inputs, train_classes)

model.predict(test_inputs)

accuracy = model.score(test_inputs, test_classes)

print("accuracy : ", accuracy * 100, "%")



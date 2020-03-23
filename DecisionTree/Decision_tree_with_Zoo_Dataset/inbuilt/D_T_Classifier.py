from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd


# True : 1
# False : 0



data = pd.DataFrame({"toothed":["1","1","1","0","1","1","1","1","1","0"],
                     "breathes":["1","1","1","1","1","1","0","1","1","1"],
                     "legs":["1","1","0","1","1","1","0","0","1","1"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]}, 
                    columns=["toothed","breathes","legs","species"])

features = data[["toothed","breathes","legs"]]
target = data["species"]


train_inputs, test_inputs , train_classes, test_classes = train_test_split(features, target, train_size = 0.8, random_state = 1)

model = DecisionTreeClassifier()

model.fit(train_inputs, train_classes)

model.predict(test_inputs)

accuracy = model.score(test_inputs, test_classes)

print("accuracy : ", accuracy * 100, "%")



'''
The dataset includes three iris species with 50 samples each as well as some properties about each flower. The available columns in this dataset are: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, and Species.

'''
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

df = load_iris()

'''
After importing the data, let’s check whether we have null values in our dataset or not.
'''

# print(df)


'''
Since there is only one dataset available (no separated training and test dataset) we need to divide the dataset into training and test dataset by ourself. To do this, we can use the train_test_split method from the scikit-learn. Don’t forget to split the input and output column to different arrays.

'''

all_inputs = df.data

target_lables = df.target

# print(all_inputs)
# print(target_lables)

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, target_lables, train_size=0.8, random_state=1)

'''
print(train_classes)
print("--------------------")
print(test_classes)
'''
'''
I configured it to split the dataset into 80:20 for training and test dataset. I also define a random_state equal to 1. The usage of defining random_state is to make sure the splitted dataset is the same even if we split the dataset again and again. It is actually only used to make sure we can reproduce the exact same dataset again.'''

'''
We will use the decision tree classifier from the scikit-learn.
'''

# print(train_classes)

# print(df.feature_names)


model = DecisionTreeClassifier()

model.fit(train_inputs, train_classes)

print("Accuracy of classifier : ",round((model.score(test_inputs, test_classes))*100,2),"%")


from sklearn.externals.six import StringIO
import pydotplus   #using pydotplus in windows10, python 3.6.X
dot_data = StringIO()

tree.export_graphviz(model, out_file=dot_data, 
                         feature_names=df.feature_names,  
                         class_names= df.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("iris.pdf")
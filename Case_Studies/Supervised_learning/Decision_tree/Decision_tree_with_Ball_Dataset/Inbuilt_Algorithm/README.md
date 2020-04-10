# Ball_Prediction Case Study

### Problem statement:

> There are multiple types of balls in our data set as cricket ball and tennis ball. This type of balls is classified based on its weight and its surface. We must design application using machine learning strategy which is used to classify the balls.

### Dataset Description

- **features ->** Weight, Surface
- **Label ->** Tennis , Cricket

### Library Installation

> pip install sklearn

### Algorithm

> DecisionTreeClassifier from sklearn

### Initial script

```
from sklearn import tree
import pickle
```

create model using DecisionTreeClassifier() class from sklearn.tree and save model using the dumps() function
```
model = tree.DecisionTreeClassifier()
saved_model = pickle.dumps(model)
```

Again load the saved model usig loads() function of pickle and call the fit() method along with our training data.

Now that we have trained our algorithm, it’s time to make some predictions. 

To do so, we will use our test data. To make predictions on the test data, execute the following script:

```
dt_from_pickle = pickle.loads(saved_model)
dt_from_pickle.fit(data,target_label)
result = dt_from_pickle.predict([[99,0]])
```

### Output

![](Output.PNG)


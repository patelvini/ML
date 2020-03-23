# Ball_Prediction Case Study

### Problem statement:

> There are multiple types of balls in our data set as cricket ball and tennis ball. This type of balls is classified based on its weight and its surface. We must design application using machine learning strategy which is used to classify the balls.

### Dataset Description

- **features ->** Weight, Surface
- **Label ->** Tennis , Cricket

### Library Installation

> pip install pandas
> pip install numpy

### Algorithm

> ID3 Algorithm for classification using Decision tree

### Initial script

```
import pandas as pd
import numpy as np
import os
import pprint
import pickle
```

create model using DecisionTreeClassifier() class from sklearn.tree and save model using the dump() function
```
model = buildTree()
saved_model = pickle.dump(model,open('filename.pkl','wb'))
```

Again load the saved model usig load() function of pickle and call the fit() method along with our training data.

Now that we have trained our algorithm, it’s time to make some predictions. 

To do so, we will use our test data. To make predictions on the test data, execute the following script:

```
dt_from_pickle = pickle.load(saved_model,open('filename.pkl','rb))
dt_from_pickle.fit(data,target_label)
result = dt_from_pickle.predict([[99,"Smooth"]])
```

### Output

![](train_model.PNG)
![](test_model.PNG)
![](predict_label.PNG)


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics


path = os.getcwd()
for i in range(3):
    path = os.path.dirname(path)
	
data = pd.read_excel( path + '/Datasets/Titanic_Dataset.xlsx')


X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived', axis=1), data['Survived'], train_size=0.7, random_state=1)

log_reg = LogisticRegression(solver='lbfgs',max_iter=500)

log_reg.fit(X_train, y_train)

parameters = log_reg.coef_
print("\nparameters : ", parameters)

predictions = log_reg.predict(X_test)

clf_report = classification_report(y_test, predictions)

print("\nclassification_report : \n\n", clf_report)

cnf_matrix = confusion_matrix(y_test,predictions)
print("Confusion matrix : \n\n",cnf_matrix)

accuracy  = round((metrics.accuracy_score(y_test, predictions))*100,3)
precision = round((metrics.precision_score(y_test, predictions))*100,3)
recall = round((metrics.recall_score(y_test, predictions))*100,3)

print("\nAccuracy : ",accuracy,"%")
print("\nPrecision : ",precision,"%")
print("\nRecall : ",recall,"%")


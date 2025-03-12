from sklearn.linear_model import LinearRegression
import os
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt

path = os.getcwd()
for i in range(3):
	path = os.path.dirname(path)
data = pd.read_csv( path + '/Datasets/HeadBrain.csv')

training_data = data.iloc[:191].reset_index(drop = True)
testing_data = data.iloc[191:].reset_index(drop = True)

# Cannot use Rank 1 matrix in scikit learn
# Reshape the input data into 2D array

X_train = training_data['Head Size(cm^3)'].values.reshape((len(training_data), 1))
Y_train = training_data['Brain Weight(grams)'].values.reshape((len(training_data), 1))

X_test = testing_data['Head Size(cm^3)'].values.reshape((len(testing_data), 1))
Y_test = testing_data['Brain Weight(grams)'].values.reshape((len(testing_data), 1))


# Creating Model
reg = LinearRegression()

filename = path + '/Model/Linear_regressor_HeadBrain.pkl'
	
pickle.dump(reg, open(filename, 'wb'))
	
# loading the saved model

reg = pickle.load(open(filename,'rb'))

# Fitting training data
reg = reg.fit(X_train, Y_train)

# print the coefficients
print("Slope : ",reg.coef_)
print("Y-intercept : " ,reg.intercept_)

# Y Prediction
Y_pred = list(reg.predict(X_test))

for i in range(len(Y_pred)):
	print(round(float(Y_pred[i]),3), end= ", ")
 
# Calculating R2 Score
print("\nR-Squared value : ",r2_score(Y_test, Y_pred))

plt.plot(X_test,Y_pred,color='red',label='Linear Regression')
plt.scatter(X_train,Y_train,c='b',label='Scatter Plot')
plt.xlabel("Head Size")
plt.ylabel("Brain Weight")
plt.legend()
plt.show() 
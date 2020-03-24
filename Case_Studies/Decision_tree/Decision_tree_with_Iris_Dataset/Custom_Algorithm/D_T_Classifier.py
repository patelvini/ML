import pandas as pd
import numpy as np
from pprint import pprint
eps = np.finfo(float).eps
import os
import pickle
import Commands as cmd

def entropy(target_col):

	"""Calculate the entropy of a dataset. The only parameter for this function is the target_col parameter which specifies the target column"""

	vals ,counts = np.unique(target_col,return_counts = True) 
	total = np.sum(counts)
	entropy = 0


	for i in counts:
		fraction = i/total
		entropy += -fraction * np.log2(fraction)

	entropy = round(entropy,3)
	return entropy

def find_entropy_attribute(df, attribute):

	Target = df.keys()[-1]

	target_variables = df[Target].unique() # This gives all 'Yes' and 'No'

	variables = df[attribute].unique()  # this gives different features of attribute 

	attribute_entropy = 0

	for variable in variables:
		entropy = 0

		for target_variable in target_variables:

			num = len(df[attribute][df[attribute] == variable][df[Target] == target_variable])

			total = len(df[attribute][df[attribute] == variable])

			fraction = num / (total+eps)

			entropy += -fraction * np.log2(fraction + eps)

		fraction_1 = total/len(df)

		attribute_entropy += -fraction_1 * entropy

	return abs(attribute_entropy)


def find_winner(df):

	attribute_entropy = []

	IG = []

	for key in df.keys()[:-1]:
		IG.append(entropy(df.keys()[-1]) - find_entropy_attribute(df,key))

	return df.keys()[:-1][np.argmax(IG)]

def get_subtable(df, node, value):

	return df[df[node] == value].reset_index(drop = True)

def buildTree(df, tree = None):

	Target = df.keys()[-1] 

	# Get attribute with maximum information gain
	node = find_winner(df)

	#Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
	att_Value = np.unique(df[node])


	# create an empty dictionary to create tree

	if tree is None:
		tree = {}
		tree[node] = {}

	# We make loop to construct a tree by calling this function recursively. 
    # In this we check if the subset is pure and stops if it is pure. 

	for value in att_Value:
		subtable = get_subtable(df, node, value)

		cl_Value, counts = np.unique(subtable[df.keys()[-1]], return_counts = True)

		if len(counts) == 1:
			tree[node][value] = cl_Value[0]  # checking purity of subset

		else:
			tree[node][value] = buildTree(subtable) # calling the function recursively

	return tree


def predict(input_data, tree):

	# this function is used to predict for any input variable

	# recursively we go through the tree that we built

	for i in tree.values():
		tree_values = i
	
	t_values = []

	for i in tree_values:
		t_values.append(float(i)) 

	for nodes in tree.keys():
		prediction = 'Verginica'
		
		value = float(input_data[nodes])

		if value in t_values:
			tree = tree[nodes][value]
			if type(tree) is dict:
				prediction = predict(input_data, tree)
			else:
				prediction = tree
				break
		else:
			break
	return prediction

def train_test_split(dataset):

	training_data = dataset.iloc[:120].reset_index(drop = True)

	testing_data = dataset.iloc[120:].reset_index(drop = True)

	return training_data, testing_data

def test(df, tree):

	queries = df.iloc[:,:-1].to_dict(orient = "records")

	predicted = pd.DataFrame(columns = ["Predicted"])


	for i in range(len(df)):
		predicted.loc[i,"predicted"] = predict(pd.Series(queries[i]),tree)

	print('The accuracy is: ',(np.sum(predicted["predicted"] == df[df.keys()[-1]])/len(df))*100,'%')


if __name__ == '__main__':

	path = os.getcwd()

	for i in range(3):
		path = os.path.dirname(path)

	data = pd.read_csv( path + '/Datasets/IRIS.csv')

	data.sort_values(by = 'sepal_width', inplace = True)

	training_data = train_test_split(data)[0]
	testing_data = train_test_split(data)[1]

	tree = buildTree(training_data)

	filename = path + '/Model/D_T_classifier_iris.pkl'
	
	pickle.dump(tree, open(filename, 'wb'))
	
	# loading the saved model

	if cmd.args.train:
		loaded_model = pickle.load(open(filename,'rb'))
		pprint(loaded_model)
	elif cmd.args.test:
		loaded_model = pickle.load(open(filename,'rb'))
		test(testing_data, loaded_model)
	elif cmd.args.predict:
		loaded_model = pickle.load(open(filename,'rb'))
		input_data = {'sepal_length': cmd.args.sepal_length, 'sepal_width' : cmd.args.sepal_width, 'petal_length': cmd.args.petal_length, 'petal_width': cmd.args.petal_width}
		input_data = pd.Series(input_data)
		prediction = predict(input_data, loaded_model)
		print(prediction)
	else:
		print("No argument given !!!")
		print(" --train for training.")
		print(" --test for check the accuracy.")
		print(" --predict for predict label for given input")
		print(" -sl for input sepal_length.")
		print(" -sw for input sepal_width.")
		print(" -pl for input petal_length.")
		print(" -pw for input petal_width.")



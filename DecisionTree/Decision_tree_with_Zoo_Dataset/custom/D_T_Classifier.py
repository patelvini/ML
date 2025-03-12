import pandas as pd
import numpy as np
from pprint import pprint

def entropy(target_col):

	"""Calculate the entropy of a dataset. The only parameter for this function is the target_col parameter which specifies the target column"""

	vals ,counts = np.unique(target_col,return_counts = True) 
	total = np.sum(counts)
	entropy = 0

	for i in range(0,len(counts)):
		entropy -= (counts[i]/total) * np.log2(counts[i]/total)

	entropy = round(entropy,3)
	return entropy

def InfoGain(data, split_attribute_name, target_name = 'species'):

	"""
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "species"
    """    

    # calculate the entropy of the total dataset
	
	total_entropy = entropy(data[target_name])

    # calculating corresponding counts for the split attribute

	vals, counts = np.unique(data[split_attribute_name], return_counts = True)

	total = np.sum(counts)
	attribute_entropy = 0


	for i in range(len(vals)):
		attribute_entropy += counts[i]/total * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])

	Information_Gain = round(total_entropy - attribute_entropy,4)

	return Information_Gain


def ID3(data, originaldata, features, target_attribute_name = 'species', parent_node_species = None):

	"""
    ID3 Algorithm: This function takes five paramters:
    1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset
 
    2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
    in the case the dataset delivered by the first parameter is empty

    3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset --> Splitting at each node

    4. target_attribute_name = the name of the target attribute

    5. parent_node_species = This is the value or species of the mode target feature value of the parent node for a specific node. This is 
    also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
    space, we want to return the mode target feature value of the direct parent node.
    """ 

    # define the stopping criteria --> If one of these is satisfied, we want to return the leaf node

	vals = np.unique(data[target_attribute_name])

    # If all target_variables have same value , return this value

	if len(vals) <= 1:
		return vals[0]

    # If the dataset is empty , return the mode target feature value in the original dataset
	
	elif len(data) == 0:

		return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts = True)[1])]

	# If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    
    # the mode target feature value is stored in the parent_node_species variable.

	elif len(features) == 0:

		return parent_node_species

	# If none of above holds true, grow the tree

	else:

		# set the default value for the node

		parent_node_species = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts = True)[1])]

		## select the feature which best splits the dataset

		item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]

		best_feature_index = np.argmax(item_values)

		best_feature = features[best_feature_index]

		# create the tree structure. The root gets the name of the feature with the maximum information Gain in the first run

		tree = {best_feature:{}}


		# Remove the feature with the best information gain from the feature space

		features = [i for i in features if i!= best_feature]

		# grow a branch under the root node for each possible value of the root node feature

		for value in np.unique(data[best_feature]):

			value = value

			# split the dataset along with the value of the feature with the largest information gain and with that create sub_dataset

			sub_data = data.where(data[best_feature] == value).dropna()

			# call the ID3 algorithm for each of those sub_datasets with the new parameters --> here the recursion comes in !!!

			subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_species)

			# Add the subtree , grown from the sub_dataset to the tree under the root node

			tree[best_feature][value] = subtree

		return tree


def predict(query, tree, default = 1):

	for key in list(query.keys()):

		if key in list(tree.keys()):

			try:
				result = tree[key][query[key]]

			except:
				return default

			result = tree[key][query[key]]

			if isinstance(result, dict):
				return predict(query,result)

			else:
				return result

def train_test_split(dataset):

	training_data = dataset.iloc[:8].reset_index(drop = True)

	testing_data = dataset.iloc[8:].reset_index(drop = True)

	return training_data, testing_data


def test(data, tree):

	queries = data.iloc[:,:-1].to_dict(orient = "records")

	predicted = pd.DataFrame(columns = ["Predicted"])


	for i in range(len(data)):
		predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)

	print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["species"])/len(data))*100,'%')





if __name__ == '__main__':

	data = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]}, 
                    columns=["toothed","breathes","legs","species"])

	features = data[["toothed","breathes","legs"]]
	target = data["species"]


	training_data = train_test_split(data)[0]
	testing_data = train_test_split(data)[1] 


	tree = ID3(data, data, data.columns[:-1])

	pprint(tree)


	tree = ID3(training_data, training_data, training_data.columns[:-1])

	pprint(tree)

	test(testing_data,tree)


# Decision Tree

### What are Decision Trees?

Decision trees are supervised learning algorithms used for both, classification and regression tasks.

Decision trees are assigned to the `information based learning algorithms` which use different measures of `information gain` for learning. We can use decision trees for issues where we have continuous but also categorical input and target features. The main idea of decision trees is to find those descriptive features which contain the most `"information"` regarding the target feature and then split the dataset along the values of these features such that the target feature values for the resulting `sub_datasets` are as pure as possible --> The descriptive feature which leaves the target feature most purely is said to be the most informative one. This process of finding the `"most informative"` feature is done until we accomplish a stopping criteria where we then finally end up in so called `leaf nodes`. The `leaf nodes` contain the predictions we will make for new query instances presented to our trained model. This is possible since the model has kind of learned the underlying structure of the training data and hence can, given some assumptions, make predictions about the target feature value (class) of unseen query instances.

A decision tree mainly contains of a root node, interior nodes, and leaf nodes which are then connected by branches.

![](https://www.python-course.eu/images/Decision_Tree1.png)

Decision trees are further subdivided whether the target feature is continuously scaled like for instance house prices or categorically scaled like for instance animal species.

![](https://www.python-course.eu/images/Categorical_Continuous_Scales.png)

In simplified terms, the process of training a decision tree and predicting the target features of query instances is as follows:

1. **Present a dataset containing of a number of training instances characterized by a number of descriptive features and a target feature**

2. **Train the decision tree model by continuously splitting the target feature along the values of the descriptive features using a measure of information gain during the training process**

3. **Grow the tree until we accomplish a stopping criteria --> create leaf nodes which represent the predictions we want to make for new query instances**

4. **Show query instances to the tree and run down the tree until we arrive at leaf nodes**

5. **DONE - Congratulations you have found the answers to your questions**

![](https://www.python-course.eu/images/Train_Predict_simplified.png)

let's take an example:

![](animal_dataset.PNG)

each leaf node should (in the best case) only contain "Mammals" or "Reptiles". The task for us is now to find the best "way" to split the dataset such that this can be achieved. What do I mean when I say split? Well consider the dataset above and think about what must be done to split the dataset into a Dataset 1 containing as target feature values (species) only Mammals and a Dataset 2, containing only Reptiles. To achieve that, in this simplified example, we only need the descriptive feature hair since if hair is TRUE, the associated species is always a Mammal. Hence in this case our tree model would look like:

![](https://www.python-course.eu/images/Reptile_Mammal_distinction.png)

That is, we have split our dataset by asking the question if the animal has hair or not. And exactly this asking and therewith splitting is the key to the decision tree models. Now in that case the splitting has been very easy because we only have a small number of descriptive features and the dataset is completely separable along the values of only one descriptive feature. However, most of the time datasets are not that easily separable and we must split the dataset more than one time .

so, let's try to split based on more descriptive features:

we have seen that using the hair descriptive feature seems to occupy the most information about the target feature since we only need this feature to perfectly split the dataset. Hence it would be useful to measure the `"informativeness"` of the features and use the feature with the most `"informativeness"` as the feature which should be used to split the data on. From now on, we use the term information gain as a measure of `"informativeness"` of a feature. In the following section we will introduce some mathematical terms and derive how the **information gain** is calculated as well as how we can build a tree model based on that.

### The maths behind Decision Trees

In the preceding section we have introduced the `information gain` as a measure of how good a descriptive feature is suited to split a dataset on. To be able to calculate the information gain, we have to first introduce the term **entropy** of a dataset. **The entropy of a dataset is used to measure the impurity of a dataset and we will use this kind of informativeness measure in our calculations.**

There are also other types of measures which can be used to calculate the information gain. 

The most prominent ones are the: 
- Gini Index
- Chi-Square
- Information gain ratio
- Variance

The term entropy (in information theory) goes back to Claude E. Shannon. The idea behind the entropy is, in simplified terms, the following: Imagine you have a lottery wheel which includes 100 green balls. The set of balls within the lottery wheel can be said to be totally pure because only green balls are included. To express this in the terminology of entropy, this set of balls has a entropy of 0 (we can also say zero impurity). Consider now, 30 of these balls are replaced by red and 20 by blue balls.

![](https://www.python-course.eu/images/Shannons_Entropy_Impurity.png)

Now our task is to find the best feature in terms of information gain (Remember that we want to find the feature which splits the data most accurate along the target feature values) which we should use to first split our data on (which serves as root node). **Remember that the hair feature is no longer part of our feature set.**

Following this, how can we check which of the descriptive features most accurately splits the dataset, that is, remains the dataset with the lowest impurity ≈ entropy or in other words best classifies the target features by its own? Well, we use each descriptive feature and split the dataset along the values of these descriptive feature and then calculate the entropy of the dataset once we have split the data along the feature values. This gives us the remaining entropy after we have split the dataset along the feature values. Next, we subtract this value from the originally calculated entropy of the dataset to see how much this feature splitting reduces the original entropy. The information gain of a feature is calculated with:

![](IG1.PNG)

So the only thing we have to do is to split the dataset along the values of each feature and then treat these sub sets as if they were our "original" dataset in terms of entropy calculation. The formula for the Information Gain calculation per feature is:

![](IG2.PNG)

Summarized, for each descriptive feature, we sum up the resulting entropies for splitting the dataset along the feature values and additionally weight the feature value entropies by their occurrence probability.

Now we will calcuate the Information gain for each descriptive feature:

**toothed:**

![](https://www.python-course.eu/images/Entropy_Calculation_of_the_toothed_feature.png)

![](https://www.python-course.eu/images/Information_Gain_Calculation.png)

![](IG_toothed.PNG)

Hence the splitting the dataset along the feature legs results in the largest information gain and we should use this feature for our root node.
Hence for the time being the decision tree model looks like:

![](https://www.python-course.eu/images/Root_Node_Split.png)

We see that for legs == False, the target feature values of the remaining dataset are all Reptile and hence we set this as leaf node because we have a pure dataset (Further splitting the dataset on any of the remaining two features would not lead to a different or more accurate result since whatever we do after this point, the prediction will remain Reptile). Additionally, you see that the feature legs is no longer included in the remaining datasets. Because we already has used this (categorical) feature to split the dataset on it must not be further used.

Until now we have found the feature for the root node as well as a leaf node for legs == False. The same steps for information gain calculation must now be accomplished also for the remaining dataset for legs == True since here we still have a mixture of different target feature values. Hence:

![](IG_toothed_1.PNG)

![](https://www.python-course.eu/images/Decision_Tree_Completed.png)

Mind the last split (node) where the dataset got split on the breathes feature. Here the breathes feature solely contains data where breaths == True. Hence for breathes == False there are no instances in the dataset and therewith there is no sub-Dataset which can be built. In that case we return the most frequently occurring target feature value in the original dataset which is Mammal. This is an example how our tree model generalizes behind the training data.
If we consider the other branch, that is breathes == True we know, that after splitting the Dataset on the values of a specific feature (breathes {True,False}) in our case, the feature must be removed. Well, that leads to a dataset where no more features are available to further split the dataset on. Hence we stop growing the tree and return the mode value of the direct parent node which is "Mammal".

![](https://www.python-course.eu/images/ID3_Pseudocode_illstration.png)






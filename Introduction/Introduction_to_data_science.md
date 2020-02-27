# Introduction to Data Science

**Data science** is a way to try and discover `hidden patterns` in `raw data`. To achieve this goal, it makes use of several algorithms, [machine learning(ML)](https://data-flair.training/blogs/machine-learning-tutorial/) principles, and scientific methods. The insights it retrieves from data lie in forms structured and unstructured. So in a way, this is like [data mining](https://data-flair.training/blogs/data-mining-tutorial/). Data science encompasses all- data analysis, statistics, and machine learning.

![](https://cdn-media-1.freecodecamp.org/images/1*ius9T3uGkd743dljInNF8w.jpeg)

### Data Science Application

- Image Recognition
- Speech Recognition
- Internet Search
- Digital Advertisements
- Recommender Systems
- Fraud and Risk Detection
- Delivery Logistics
- Price Comparison Websites

# Concept of Artificial Intelligence

![](https://www.nopsec.com/wp-content/uploads/Artificial-Intelligence-Machine-Learning-Deep-Learning3.png)

### what is Artificial Intelligence ?

![](https://www.datamation.com/imagesvr_ce/9138/Artificial-Intelligence.png)

Artificial intelligence (AI) is an area of computer science that emphasizes the creation of intelligent machines that work and react like humans. Some of the activities computers with  

The core problems of artificial intelligence include programming computers for certain traits such as:
- Knowledge
- Reasoning
- Problem solving
- Perception
- Learning
- Planning
- Ability to manipulate and move objects

### AI in the Real World
There is no shortage of compelling use cases for AI. Here are some leading examples:

##### Healthcare
Artificial intelligence in healthcare can play a leading role. It enables health professionals to understand risk factors and diseases at a deeper level. It can aid in diagnosis and provide insight into risks. AI also powers smart devices, surgical robots and Internet of Things (IoT) systems that support patient tracking or alerts.

##### Agriculture
AI is now widely used for crop monitoring. It helps farmers apply water, fertilizer and other substances at optimal levels. It also aids in preventative maintenance for farm equipment and it is spawning autonomous robots that pick crops.

##### Finance
Few industries have been transformed by AI more than finance. Today, quants (algorithms) trade stocks with no human intervention, banks make automated credit decisions instantly, and financial organizations use algorithms to spot fraud. AI also allows consumers to scan paper checks and make deposits using a smartphone.

##### Retail
A growing number of consumer-facing apps and tools support image recognition, voice and natural language processing and augmented reality (AR) features that allow consumers to preview a piece of furniture in a room or office or see what makeup looks like without heading to a physical store. Retailers are also using AI for personalized marketing, managing supply chains, and cybersecurity.

##### Travel, Transportation and Hospitality
Airlines, hotels, and rental car companies use AI to forecast demand and adapt pricing dynamically. Airlines also rely on AI to optimize the use of aircraft for routes, factoring in weather conditions, passenger loads and other variables. They can also understand when aircraft require maintenance. Hotels are using AI, including image recognition, for deploying robots and security monitoring. Autonomous vehicles and smart transportation grids also rely on AI.

# Concept of Machine Learning

### What is Machine Learning?
Machine Learning is the most popular technique of predicting the future or classifying information to help people in making necessary decisions. Machine Learning algorithms are trained over instances or examples through which they learn from past experiences and also analyze the historical data. Therefore, as it trains over the examples, again and again, it is able to identify patterns in order to make predictions about the future.

![](https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2017/07/what-is-machine-learning.jpg)

### How does Machine Learning Work?
With an exponential increase in data, there is a need for having a system that can handle this massive load of data. Machine Learning models like Deep Learning allow the vast majority of data to be handled with an accurate generation of predictions. Machine Learning has revolutionized the way we perceive information and the various insights we can gain out of it.

![](https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/07/How-machine-learning-works.jpg)

These machine learning algorithms use the patterns contained in the training data to perform classification and future predictions. Whenever any new input is introduced to the ML model, it applies its learned patterns over the new data to make future predictions. Based on the final accuracy, one can optimize their models using various standardized approaches. In this way, Machine Learning model learns to adapt to new examples and produce better results.

#### What is the machine learning Model?
The machine learning model is nothing but a piece of code; an engineer or data scientist makes it smart through training with data. So, if you give garbage to the model, you will get garbage in return, i.e. the trained model will provide false or wrong predictions.


# Developmental phases of Machine Learning

We can define the machine learning workflow in 5 stages.
1. Gathering data
2. Data pre-processing
3. Researching the model that will be best for the type of data
4. Training and testing the model
5. Evaluation

#### Gathering Data
The process of gathering data depends on the type of project we desire to make, if we want to make an ML project that uses real-time data, then we can build an IoT system that using different sensors data. The data set can be collected from various sources such as a file, database, sensor and many other such sources but the collected data cannot be used directly for performing the analysis process as there might be a lot of missing data, extremely large values, unorganized text data or noisy data. Therefore, to solve this problem Data Preparation is done.

![](https://miro.medium.com/max/846/1*0dJNCFj1hjLjmsqBhJ0_2w.png)

![](https://miro.medium.com/max/384/1*z55Hi1HtLpdBhhPQbS9rZQ.gif)

![](https://miro.medium.com/max/770/1*gh-vDoSQEWrl_KgLjybYXA.jpeg)

We can also use some free data sets which are present on the internet. [Kaggle](https://www.kaggle.com/) and UCI Machine learning Repository are the repositories that are used the most for making Machine learning models. Kaggle is one of the most visited websites that is used for practicing machine learning algorithms, they also host competitions in which people can participate and get to test their knowledge of machine learning.

#### Data pre-processing
Data pre-processing is one of the most important steps in machine learning. It is the most important step that helps in building machine learning models more accurately. In machine learning, there is an 80/20 rule. Every data scientist should spend 80% time for data pre-processing and 20% time to actually perform the analysis.

##### What is data pre-processing?
Data pre-processing is a process of cleaning the raw data i.e. the data is collected in the real world and is converted to a clean data set. In other words, whenever the data is gathered from different sources it is collected in a raw format and this data isn’t feasible for the analysis.

Therefore, certain steps are executed to convert the data into a small clean data set, this part of the process is called as data pre-processing.

##### Why do we need it?
As we know that data pre-processing is a process of cleaning the raw data into clean data, so that can be used to train the model. So, we definitely need data pre-processing to achieve good results from the applied model in machine learning and deep learning projects.

Most of the real-world data is messy, some of these types of data are:
1. **Missing data:** Missing data can be found when it is not continuously created or due to technical issues in the application (IOT system).
2. **Noisy data:** This type of data is also called outliners, this can occur due to human errors (human manually gathering the data) or some technical problem of the device at the time of collection of data.
3. **Inconsistent data:** This type of data might be collected due to human errors (mistakes with the name or values) or duplication of data.

##### Three Types of Data
1. Numeric e.g. income, age
2. Categorical e.g. gender, nationality
3. Ordinal e.g. low/medium/high

##### How can data pre-processing be performed?
These are some of the basic pre — processing techniques that can be used to convert raw data.
1. **Conversion of data:** As we know that Machine Learning models can only handle numeric features, hence categorical and ordinal data must be somehow converted into numeric features.
2. **Ignoring the missing values:** Whenever we encounter missing data in the data set then we can remove the row or column of data depending on our need. This method is known to be efficient but it shouldn’t be performed if there are a lot of missing values in the dataset.
3. **Filling the missing values:** Whenever we encounter missing data in the data set then we can fill the missing data manually, most commonly the mean, median or highest frequency value is used.
4. **Machine learning:** If we have some missing data then we can predict what data shall be present at the empty position by using the existing data.
5. **Outliers detection:*** There are some error data that might be present in our data set that deviates drastically from other observations in a data set. [Example: human weight = 800 Kg; due to mistyping of extra 0]


#### Researching the model that will be best for the type of data
Our main goal is to train the best performing model possible, using the pre-processed data.

#### Training and testing the model on data
For training a model we initially split the model into 3 three sections which are ‘Training data’ ,‘Validation data’ and ‘Testing data’.
You train the classifier using ‘training data set’, tune the parameters using ‘validation set’ and then test the performance of your classifier on unseen ‘test data set’. An important point to note is that during training the classifier only the training and/or validation set is available. The test data set must not be used during training the classifier. The test set will only be available during testing the classifier.

![](https://miro.medium.com/max/850/1*kpqurK-46RQxCllffLgM3w.png)

#### Evaluation
Model Evaluation is an integral part of the model development process. It helps to find the best model that represents our data and how well the chosen model will work in the future.

![](https://miro.medium.com/max/711/1*_0KYNHYB3DokiqfJWKvAGw.png)

To improve the model we might tune the hyper-parameters of the model and try to improve the accuracy and also looking at the confusion matrix to try to increase the number of true positives and true negatives.



import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns 


iris = datasets.load_iris()


print(iris.feature_names)
print(iris.target_names)

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']).values
y = pd.DataFrame(iris.target, columns=['Target']).values

# Elobow method for finding no. of k

wcss = []

for i in range(1, 11):
	k_mean = KMeans(n_clusters = i, init='k-means++', random_state = 42)
	k_mean.fit(x)

	wcss.append(k_mean.inertia_)

# plot graph for elbow point

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
exit()


k_mean = KMeans(n_clusters = 3)

labels = k_mean.fit_predict(x)

centroids = k_mean.cluster_centers_

print("\nCentroids : \n",centroids)

print("\nPredicted labels : \n",labels)

# Visualising the clusters
plt.scatter(x[labels == 0, 0], x[labels == 0,1],s=100,c='red',label='Setosa')

plt.scatter(x[labels == 1, 0], x[labels == 1,1],s=100,c='blue',label='Versicolor')

plt.scatter(x[labels == 2, 0], x[labels == 2,1],s=100,c='green',label='Virginica')

plt.scatter(centroids[:,0],centroids[:,1],s=200,c='black',label='Centroids',marker="*")

plt.title('Clusters for iris flowers')
plt.legend()
plt.show()



































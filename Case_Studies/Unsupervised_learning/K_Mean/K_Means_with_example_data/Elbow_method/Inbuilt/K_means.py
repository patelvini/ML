import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans

X1 = np.array([12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72])
Y1 = np.array([39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24])


X = np.array(list(zip(X1, Y1))).reshape(len(X1), 2)


# Elobow method for finding no. of k

wcss = []

for i in range(1, 11):
	k_mean = KMeans(n_clusters = i, init='k-means++', random_state = 42)
	k_mean.fit(X)

	wcss.append(k_mean.inertia_)

# plot graph for elbow point

# plt.figure(figsize=(10,5))
# sns.lineplot(range(1, 11), wcss,marker='o',color='red')
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

k_mean = KMeans(n_clusters = 3)
k_mean.fit(X)

centroids = k_mean.cluster_centers_

print("\nCentroids : \n",centroids)

# visualization of clustered data

plt.plot()
plt.title('k means centroids')
plt.xlabel('X')
plt.ylabel('Y')

print("\nLabels : \n",k_mean.labels_)

plt.scatter(X[:, 0], X[:, 1], c=k_mean.labels_ )

plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
plt.show()
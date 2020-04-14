import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from sklearn import datasets
import pandas as pd


class KMeans:

	def __init__(self, n_clusters, max_iter=100, random_state = 123):

		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.random_state = random_state

	def initialize_centroids(self, X):
		np.random.RandomState(self.random_state)
		random_idx = np.random.permutation(X.shape[0])
		centroids = X[random_idx[:self.n_clusters]]
		return centroids

	def compute_centroids(self, X, labels):
		centroids = np.zeros((self.n_clusters, X.shape[1]))
		for k in range(self.n_clusters):
			centroids[k, :] = np.mean(X[labels == k, :], axis=0)
		return centroids

	def compute_distance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			row_norm = norm(X - centroids[k, :], axis=1)
			distance[:, k] = np.square(row_norm)
		return distance

	def find_closest_cluster(self, distance):
		return np.argmin(distance, axis=1)

	def compute_sse(self, X, labels, centroids):
		distance = np.zeros(X.shape[0])
		for k in range(self.n_clusters):
			distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
		return np.sum(np.square(distance))
    
	def fit(self, X):
		self.centroids = self.initialize_centroids(X)
		for i in range(self.max_iter):
			old_centroids = self.centroids
			distance = self.compute_distance(X, old_centroids)
			self.labels = self.find_closest_cluster(distance)
			self.centroids = self.compute_centroids(X, self.labels)
			if np.all(old_centroids == self.centroids):
				break
			self.error = self.compute_sse(X, self.labels, self.centroids)

		return self.centroids,self.labels
    
	def predict(self, X):
		self.centroids = self.initialize_centroids(X)
		old_centroids = self.centroids
		distance = self.compute_distance(X, old_centroids)
		return self.find_closest_cluster(distance)


if __name__ == '__main__':

	iris = datasets.load_iris()

	x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']).values
	y = pd.DataFrame(iris.target, columns=['Target']).values

	k_mean = KMeans(3)

	initial_centroids = k_mean.initialize_centroids(x)
	# Visualising the clusters
	# plt.scatter(x[:,0], x[:,1],s=100,c='blue')

	# plt.scatter(initial_centroids[:,0],initial_centroids[:,1],s=200,c='black',marker="*")

	# plt.title('Initial centroids')
	# plt.legend()
	# plt.show()

	# exit()

	centroids, labels = k_mean.fit(x)
	
	print("\nCentroids : \n",centroids)

	print("\nLabels : \n",labels)

	# visualization of clustered data

	# Visualising the clusters
	plt.scatter(x[labels == 0, 0], x[labels == 0,1],s=100,c='red',label='Setosa')

	plt.scatter(x[labels == 1, 0], x[labels == 1,1],s=100,c='blue',label='Versicolor')

	plt.scatter(x[labels == 2, 0], x[labels == 2,1],s=100,c='green',label='Virginica')

	plt.scatter(centroids[:,0],centroids[:,1],s=200,c='black',label='Centroids',marker="*")

	plt.title('Clusters for iris flowers')
	plt.legend()
	plt.show()






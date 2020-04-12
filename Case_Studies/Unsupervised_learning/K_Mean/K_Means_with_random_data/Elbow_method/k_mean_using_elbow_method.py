import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from random import randint
from sklearn.cluster import KMeans
import seaborn as sns

class Kmeans:

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

	X1 = np.random.randint(100,size=300)
	Y1 = np.random.randint(100, size=300)

	X = np.array(list(zip(X1, Y1))).reshape(len(X1), 2)

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

	# exit()


	k_mean = Kmeans(4)

	initial_centroids = k_mean.initialize_centroids(X)
	# plt.scatter(X[:, 0], X[:, 1] )
	# plt.scatter(initial_centroids[:,0], initial_centroids[:,1], marker="x", color='r')

	# plt.show()
	# exit()

	centroids, labels = k_mean.fit(X)
	
	print("\nCentroids : \n",centroids)

	print("\nLabels : \n",labels)

	# visualization of clustered data

	plt.plot()
	plt.title('k means centroids')
	plt.xlabel('X')
	plt.ylabel('Y')

	plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

	plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
	plt.show()
	





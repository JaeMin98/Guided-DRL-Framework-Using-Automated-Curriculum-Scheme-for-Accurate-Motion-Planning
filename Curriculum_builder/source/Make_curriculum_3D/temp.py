import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

# Sample data for demonstration
X = np.random.rand(100, 2)

# Creating and fitting the Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predicting the cluster assignments for the data points
labels = gmm.predict(X)

# Calculating cluster centers (mean of data points within each cluster)
cluster_centers = np.array([X[labels == k].mean(axis=0) for k in range(gmm.n_components)])
print(cluster_centers)
# Plotting the data points
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model Clustering')
plt.legend()
plt.show()

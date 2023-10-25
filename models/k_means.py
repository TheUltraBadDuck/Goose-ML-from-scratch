import numpy as np

from models.model import UnsupervisedModel



class KMeansClustering(UnsupervisedModel):

    def __init__(self, k=5, iters=100, plot_steps=False):
        self.k = k
        self.iters = iters
        self.plot_steps = plot_steps


    def fit(self, X):
        # Choose random positions
        center_i_list = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[center_i_list, :]

        self.cluster_list = np.zeros(X.shape[0])
        for iter in range(1):
            # Make a list of clusters
            # By choosing the index of the row that has the shortest distance to each group
            for i in range(self.cluster_list.shape[0]):
                min_dist = 32768.0
                for center_k, center_v in enumerate(self.centroids):
                    dist = np.linalg.norm(center_v - X[i, :])
                    if min_dist > dist:
                        min_dist = dist
                        self.cluster_list[i] = center_k
                
            # Calculate the average value of each cluster
            for group in range(self.k):
                self.centroids[group] = np.mean(X[self.cluster_list == group])


    def transform(self, X):
        return self.cluster_list


    def makePlot(self, X, y, X_transform, ax, legends):
        colors = ["#291D25", "#8A7090", "#823167", "#9D4BA2", "#FACFEC"]
        for group in range(self.k):
            ax.scatter(x=X[X_transform == group, 0],
                       y=X[X_transform == group, 1],
                       color=colors[group],
                       marker='o', linewidths=0)
            



import numpy as np
from matplotlib.colors import ListedColormap

from models.model import UnsupervisedModel



class PCA(UnsupervisedModel):

    def __init__(self, n_components):
        self.n_components = n_components


    def fit(self, X):

        X_update = np.copy(X)
        # mean
        self.mean = np.mean(X, axis=0)
        X_update = X_update - self.mean

        # covariance
        cov = np.cov(X_update.T)

        # eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]


    def transform(self, X):
        X_update = X - self.mean
        return np.dot(X_update, self.components.T)



    def makePlot(self, X, y, X_transform, ax, legends):
        x1 = X_transform[:, 0]
        x2 = X_transform[:, 1]

        scatter = ax.scatter(x1, x2, c=y,
                                edgecolor="none",
                                alpha=0.8,
                                cmap=ListedColormap(["#8D3B72", "#8A7090", "#89A7A7"]))

        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")
        ax.set_xlabel("Principal Component 1", fontsize=14)
        ax.set_ylabel("Principal Component 2", fontsize=14)

        if len(legends) > 0:
            handles = scatter.legend_elements(num=len(legends))[0]
            ax.legend(handles=handles, labels=legends, ncols=len(legends), loc="lower center")



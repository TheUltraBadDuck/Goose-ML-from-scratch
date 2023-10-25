import numpy as np

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



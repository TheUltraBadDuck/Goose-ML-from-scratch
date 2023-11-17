import numpy as np
from collections import Counter
from matplotlib.colors import ListedColormap

from models.model import SupervisedModel
from tools.data_base import GlassColourList



def getEuclideanDistance(x1, x2):
    return np.sqrt(np.sum(np.subtract(x1, x2) ** 2))



class KNN(SupervisedModel):
    
    def __init__(self, k: int = 3):
        self.k = k



    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train



    def predict(self, X_test):
        predicted_labels = [self._predict(x) for x in X_test]
        return np.array(predicted_labels)



    def _predict(self, row):
        distances = [getEuclideanDistance(row, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



    def makePlot(self, X_test, y_test, y_pred, x, y, ax):
        for i in range(y_pred.shape[0]):
            if y_pred[i] != y_test[i]:
                ax.scatter(X_test[i, x], X_test[i, y], s=120, c="#FFF0", edgecolors="#7E2743", linewidths=2)
        



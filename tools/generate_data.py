import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from tools.tool import Tool



class GenerateData(Tool):

    def __init__(self, type: str = "", n_samples: int = 100, n_features: int = 2, centers: int = 2):
        # Load Iris dataset
        super().__init__()
        if len(type) > 0:
            self.loadData(type, n_samples, n_features, centers)
    


    def loadData(self, type: str, n_samples: int, n_features: int, centers: int):
        self.type = type

        match type:
            case "regression":
                self.X, self.y = datasets.make_regression(n_samples=n_samples,
                                                          n_features=n_features,
                                                          noise=10, random_state=4)
                self.title = "Linear Regression Models"
                self.limit = [0]

            case "blobs":
                self.X, self.y = datasets.make_blobs(n_samples=n_samples,
                                                     n_features=n_features,
                                                     centers=centers, cluster_std=1.05, random_state=2)
                self.title = "Clustering Models"
                self.limit = [0, 1]

            case _:
                raise Exception("Need to load a property dataset, or this dataset is not available yet.")

    


    def checkAccuracy(self):
        fig, ax = plt.subplots()

        match self.type:

            case "regression":
                ax.scatter(x=self.X_test[:, 0],
                           y=self.y_test,
                           color="#8D3B72",
                           marker='o', linewidths=0)
                ax.scatter(x=self.X_test[:, 0],
                           y=self.y_pred,
                           color="#72e1d1",
                           marker='o', linewidths=0)
            
            case "blobs":
                mask_arr = np.where(self.y_test == 0, False, True)
                ax.scatter(x=self.X_test[mask_arr, 0],
                           y=self.X_test[mask_arr, 1],
                           color="#8D3B72",
                           marker='o', linewidths=0)
                ax.scatter(x=self.X_test[np.logical_not(mask_arr), 0],
                           y=self.X_test[np.logical_not(mask_arr), 1],
                           color="#8a7090",
                           marker='o', linewidths=0)
                self.model.makePlot(self.X_test, ax)
                
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle("Linear data", fontsize=16, fontweight='bold')

        plt.show()



    def draw(self):
        fig, ax = plt.subplots()

        match self.type:

            case "regression":
                ax.scatter(x=self.X_train[:, 0],
                           y=self.y_train,
                           color="#8D3B72",
                           marker='o', linewidths=0)
                
            case "blobs":
                mask_arr = np.where(self.y_train == 0, False, True)
                ax.scatter(x=self.X_train[mask_arr, 0],
                           y=self.X_train[mask_arr, 1],
                           color="#8D3B72",
                           marker='o', linewidths=0)
                ax.scatter(x=self.X_train[np.logical_not(mask_arr), 0],
                           y=self.X_train[np.logical_not(mask_arr), 1],
                           color="#8a7090",
                           marker='o', linewidths=0)
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        plt.show()



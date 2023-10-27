import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.model import SupervisedModel, UnsupervisedModel

from tools.tool import Tool



class GenerateData(Tool):

    def __init__(self, data_type: str = "",
                 n_samples: int = 100, n_features: int = 2, centers: int = 2,
                 noise=10, random_state=4):
        # Load Iris dataset
        super().__init__()
        if len(data_type) > 0:
            self.loadData(data_type, n_samples, n_features, centers, noise, random_state)
    


    def loadData(self, data_type: str, n_samples: int, n_features: int, centers: int, noise=10, random_state=4):
        self.type = data_type

        match data_type:
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

        if issubclass(type(self.model), SupervisedModel):
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
        
        else:
            self.model.makePlot(self.X, self.y, self.X_transform, ax, "")
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle("Linear data", fontsize=16, fontweight='bold')

        plt.show()



    def draw(self):
        fig, ax = plt.subplots()

        if issubclass(type(self.model), SupervisedModel):
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



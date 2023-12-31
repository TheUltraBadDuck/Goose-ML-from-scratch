import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from models.model import SupervisedModel
from tools.data_base import DataBase



class DataLoader(DataBase):

    def __init__(self, data_type: str = "", limit: list = []):
        super().__init__()
        if len(data_type) > 0:
            self.loadData(data_type, limit)
    


    def loadData(self, data_type: str, limit: list):
        match data_type:
            case "iris":
                ds = datasets.load_iris()
                self.title = "Iris Dataset"
                self.feature_titles = ("Sepal length", "Sepal width", "Petal length", "Petal width")
                self.legends = ("Setosa", "Versicolor", "Virginica")

            case "breast_cancer":
                ds = datasets.load_breast_cancer()
                self.title = "Breast Cancer Dataset"
                self.feature_titles = (
                    "Mean Radius",     "Mean Texture",     "Mean Area",      "Mean Perimeter",      "Mean Asymmetry",
                    "Mean Smoothness", "Mean Compactness", "Mean Concavity", "Mean Concave points", "Mean Symmetry",
                    "STD Error Mean Radius",     "STD Error Mean Texture",     "STD Error Mean Area",      "STD Error Mean Perimeter",      "STD Error Mean Asymmetry",
                    "STD Error Mean Smoothness", "STD Error Mean Compactness", "STD Error Mean Concavity", "STD Error Mean Concave points", "STD Error Mean Symmetry",
                    "Worst Radius",     "Worst Texture",     "Worst Area",      "Worst Perimeter",      "Worst Asymmetry",
                    "Worst Smoothness", "Worst Compactness", "Worst Concavity", "Worst Concave points", "Worst Symmetry")
                self.legends = ("Benign", "Malignant")
            
            case "wine":
                ds = datasets.load_wine()
                self.title = "Wine Dataset"
                self.feature_titles = (
                    "Alcohol", "Malic.acid", "Ash", "Acl", "Mg", "Phenols",
                    "Flavanoids", "Nonflavanoid.phenols", "Proanth", "Color.int", "Hue", "OD", "Proline"
                )
                self.legends = ("Bad", "Good", "Excellent")

            case _:
                raise Exception("Need to load a property dataset, or this dataset is not available yet.")

        self.X = ds.data
        self.y = ds.target
        print(self.y)
        self.limit = limit if len(limit) != 0 else [0, 1, 2, 3]
        
    

    def drawTrain(self, feature: int = 0):
        fig, ax = plt.subplots()

        ax.scatter(x=self.X_train[:, feature],
                   y=self.y_train,
                   color="#8D3B72",
                   marker='o', linewidths=0)
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        plt.show()



    def compare(self):
        list_len = np.clip(len(self.limit), 0, 4)
        fig, ax = plt.subplots(list_len, list_len)

        for i in range(list_len):
            for j in range(list_len):

                if i == j:
                    ax[i, j].text(0.5, 0.5, self.feature_titles[self.limit[i]], fontsize=12, fontstyle='italic', va="center", ha="center")
                
                else:
                    scatter = ax[i, j].scatter(x=self.X[:, j],
                                               y=self.X[:, i],
                                               c=self.y, cmap=ListedColormap(["#8D3B72", "#8A7090", "#89A7A7"]),
                                               marker='o', linewidths=0, s=5*2**(5-list_len))

                ax[i, j].get_xaxis().set_ticks([])
                ax[i, j].get_yaxis().set_ticks([])
                for pos in ['top', 'bottom', 'right', 'left']:
                    ax[i, j].spines[pos].set_edgecolor("#B5D8CC")


        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        if len(self.legends) > 0:
            handles = scatter.legend_elements(num=len(self.legends))[0]
            fig.legend(handles=handles, labels=self.legends, ncols=len(self.legends), loc="lower center")

        plt.show()
    


    def drawTest(self, feature: int = 0, graph_type="scatter", feature_x: int = 0, feature_y: int = 1):

        fig, ax = plt.subplots()

        if issubclass(type(self.model), SupervisedModel):
            match graph_type:

                case "scatter":
                    ax.scatter(x=self.X_test[:, feature],
                            y=self.y_test,
                            s=16,
                            color="#8D3B72",
                            marker='o', linewidths=0)

                    ax.scatter(x=self.X_test[:, feature],
                            y=self.y_pred,
                            s=12,
                            color="#72e1d1",
                            marker='H', linewidths=0)
                    
                    ax.tick_params(axis='x', colors="#8a7090")
                    ax.tick_params(axis='y', colors="#8a7090")
                    for pos in ['top', 'bottom', 'right', 'left']:
                        ax.spines[pos].set_edgecolor("#B5D8CC")
                
                case "heatmap":
                    legend_len = len(self.legends)
                    iris_matrix = np.zeros((legend_len, legend_len), dtype=int)
                    for i in range(len(self.y_pred)):
                        iris_matrix[self.y_test[i], self.y_pred[i]] += 1

                    ax.imshow(iris_matrix, cmap=LinearSegmentedColormap.from_list("", ["#FFFFFF", "#72E1D1"]))

                    ax.set_xticks(np.arange(legend_len), labels=self.legends)
                    ax.set_yticks(np.arange(legend_len), labels=self.legends)
                    ax.set_xlabel("Predicted values", fontsize=14)
                    ax.set_ylabel("Actual values"   , fontsize=14)
                    for pos in ['top', 'bottom', 'right', 'left']:
                        ax.spines[pos].set_edgecolor("#B5D8CC")
                    
                    for i in range(legend_len):
                        for j in range(legend_len):
                            ax.text(j, i, iris_matrix[i, j], fontsize=16, ha="center", va="center")
                
                case "region":
                    print(self.y_test)
                    print(self.y_pred)
                    self.model.makePlot(self.X_test, ax, "TTTT", [feature_x, feature_y])
                    ax.scatter(self.X_test[:, feature_x],
                               self.X_test[:, feature_y],
                               c=self.y_test, s=50,
                               cmap=ListedColormap(["#291D25", "#724B5F", "#823167", "#9D4BA2", "#FACFEC"]),
                               linewidths=1,
                               edgecolors="gray")

        else:
            self.model.makePlot(self.X, self.X_transform, ax, self.legends)



        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        plt.show()



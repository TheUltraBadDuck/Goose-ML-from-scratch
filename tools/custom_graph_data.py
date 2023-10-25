import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from tools.tool import Tool



class CustomGraphData(Tool):

    def __init__(self):
        super().__init__()
    


    def loadData(self, func, min_x: int, min_y: int, quantity: int, noise=20):
        self.func = func
        self.X = np.linspace(min_x, min_y, quantity, dtype=float)
        self.X = np.reshape(self.X, (-1, 1))
        self.y = np.array([func(x[0]) for x in self.X])
        self.y += np.random.normal(0, noise, size=(self.X.shape[0]))
        self.y = np.reshape(self.y, (-1, 1))
        
    


    def checkAccuracy(self, feature: int = 0):
        
        fig, ax = plt.subplots()

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

        self.model.makePlot(self.X_test, self.y_pred, ax)
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle("Graph test", fontsize=16, fontweight='bold')
        plt.show()




    def train(self):
        self.model.fit(self.X_train, self.y_train)



    def test(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred



    def draw(self, feature: int = 0):
        fig, ax = plt.subplots()

        ax.scatter(x=self.X_train[:, feature],
                   y=self.y_train,
                   color="#8D3B72",
                   marker='o', linewidths=0)
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle("Linear data", fontsize=16, fontweight='bold')
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
                                               marker='o', linewidths=0)

                ax[i, j].get_xaxis().set_ticks([])
                ax[i, j].get_yaxis().set_ticks([])
                for pos in ['top', 'bottom', 'right', 'left']:
                    ax[i, j].spines[pos].set_edgecolor("#B5D8CC")


        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        if len(self.legends) > 0:
            handles = scatter.legend_elements(num=len(self.legends))[0]
            fig.legend(handles=handles, labels=self.legends, ncols=len(self.legends), loc="lower center")

        plt.show()



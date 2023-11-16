from matplotlib.colors import ListedColormap
import numpy as np

from models.model import SupervisedModel
from tools.activ_func import Sigmoid, Softmax



class SoftmaxRegression(SupervisedModel):

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 100, optimized_softmax: bool = False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activ_func = lambda x: Softmax(x, optimized_softmax)



    @staticmethod
    def encodeOneHot(y):
        unique = np.unique(y)
        # # Check if one hot is practical
        # if unique.shape[0] > y.shape[0] / 10:
        #     raise Exception("Unable to encode one hot because the data is not suitable for classification")
        y_onehot = np.zeros((y.shape[0], unique.shape[0]))
        y_onehot[np.arange(y.shape[0]), y] = 1
        
        return y_onehot, unique



    def fit(self, X_train, y_train):
        y_onehot, unique = self.encodeOneHot(y_train)

        self.w = np.zeros((X_train.shape[1], unique.shape[0]))
        self.b = np.zeros((1, unique.shape[0]))

        for iter in range(self.n_iters):
            z_val = np.dot(X_train, self.w) + self.b
            pred_y = self.activ_func(z_val)
            
            error =  pred_y - y_onehot
            dw = 1 / X_train.shape[0] * np.dot(X_train.T, error)
            db = 1 / X_train.shape[0] * np.mean(error, axis=0)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db



    def predict(self, X_test):
        z_val = np.dot(X_test, self.w) + self.b
        y_pred = self.activ_func(z_val)
        return np.argmax(y_pred, axis=1)



    def makePlot(self, X_test, ax, title, feature):

        # Create a meshgrid of points over the input space.
        x_min, x_max = X_test[:, feature[0]].min() - 0.1, X_test[:, feature[0]].max() + 0.05
        y_min, y_max = X_test[:, feature[1]].min() - 0.1, X_test[:, feature[1]].max() + 0.05
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # Predict the class of each point in the meshgrid.
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.reshape(Z, (xx.shape[0], -1))

        # Color the points in the meshgrid according to their predicted class.
        colors = ['red', 'green', 'blue']
        cmap = ListedColormap(["#291D2566", "#8A709066", "#82316766", "#9D4BA266", "#FACFEC66"])
        Z_colors = cmap(Z)

        # Plot the meshgrid.
        ax.pcolormesh(xx, yy, Z_colors, shading='auto')

        # Set the axis labels and title.
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)



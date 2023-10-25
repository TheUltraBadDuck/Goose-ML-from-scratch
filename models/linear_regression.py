import numpy as np

from models.model import SupervisedModel



class LinearRegression(SupervisedModel):

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters



    def fit(self, X_train, y_train):
        self.w = np.zeros(X_train.shape[1])
        self.b = 0
        
        for iter in range(self.n_iters):
            pred_y = np.dot(X_train, self.w) + self.b
            dw = 1 / X_train.shape[0] * np.dot(X_train.T, np.subtract(pred_y, y_train))
            db = 1 / X_train.shape[0] * np.sum(np.subtract(pred_y, y_train))
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db



    def predict(self, X_test):
        y_pred = np.add(np.dot(X_test, self.w), self.b)
        return y_pred



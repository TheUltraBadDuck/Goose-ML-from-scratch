import numpy as np

from models.model import SupervisedModel



class SVM(SupervisedModel):

    def __init__(self, learning_rate: float = 0.01, lambda_param=0.01, n_iters: int = 100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param



    def fit(self, X_train, y_train):
        y_sign = np.where(y_train <= 0, -1, 1)

        self.w = np.zeros(X_train.shape[1])
        self.b = 0

        for iter in range(self.n_iters):
            for k, v in enumerate(X_train):
                z = np.dot(v, self.w) + self.b
                condition = (y_sign[k] * z) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    self.w -= self.learning_rate * dw
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(v, y_sign[k])
                    self.w -= self.learning_rate * dw
                    self.b -= self.learning_rate * (-y_sign[k])



    def predict(self, X_test):
        linear_output = np.dot(X_test, self.w) + self.b
        return np.where(linear_output <= 0, 0, 1)



    def makePlot(self, X_test, ax, title, feature):
        
        x_0_start = np.min(X_test[:, 0])
        x_1_start = np.max(X_test[:, 0])

        x_0_end = -(x_0_start * self.w[0] + self.b) / self.w[1]
        x_1_end = -(x_1_start * self.w[0] + self.b) / self.w[1]

        x_0_pos = -(x_0_start * self.w[0] + self.b - 1) / self.w[1]
        x_1_pos = -(x_1_start * self.w[0] + self.b - 1) / self.w[1]

        x_0_neg = -(x_0_start * self.w[0] + self.b + 1) / self.w[1]
        x_1_neg = -(x_1_start * self.w[0] + self.b + 1) / self.w[1]

        ax.plot([x_0_start, x_1_start], [x_0_end,   x_1_end], linewidth=2, color="#72E1D1")
        ax.plot([x_0_start, x_1_start], [x_0_pos,   x_1_pos], linewidth=2, color="#B5D8CC", linestyle="dashed")
        ax.plot([x_0_start, x_1_start], [x_0_neg,   x_1_neg], linewidth=2, color="#B5D8CC", linestyle="dashed")




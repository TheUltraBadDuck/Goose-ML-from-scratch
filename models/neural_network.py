import numpy as np

from models.model import SupervisedModel
from tools.activ_func import *



def RMSE(y, y_hat):
    dist: np = y - y_hat
    total: float = np.sum(np.multiply(dist, dist))
    return np.sqrt(total / y.shape[0])



class NeuralNetwork(SupervisedModel):

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 100,
                 hidden_nodes: list = [2], activ_func = [ReLU, Linear], deriv_func = [ReLU_Deriv, Linear_Deriv],
                 running_time=1, printing=False):
        
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.hidden_nodes = hidden_nodes
        self.activ_func = activ_func
        self.deriv_func = deriv_func
        self.running_time = running_time
        self.printing = printing



    def fit(self, X_train, y_train):

        min_error = 32768.0
        self.w, self.b = None, None

        for running in range(self.running_time):

            out = len(self.activ_func)

            # A list of weights and bias
            w = [None for _ in range(out)]
            b = [None for _ in range(out)]
            z = [None for _ in range(out)]
            y = [None for _ in range(out + 1)]

            s_iter = [ X_train.shape[1], *[ self.hidden_nodes[i] for i in range(len(self.hidden_nodes)) ], 1 ]
            for i in range(out):
                w[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
                b[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))


            # Train
            y[0] = X_train
            prev_error = 0.0
            for iter in range(self.n_iters):

                # Forward
                for k in range(out):
                    z[k]     = np.dot(y[k], w[k]) + b[k]
                    y[k + 1] = self.activ_func[k](z[k])

                error = y[out] - np.reshape(y_train, (-1, 1))
                dw = [None for _ in range(out)]
                db = [None for _ in range(out)]
                
                curr_error = RMSE(y_train, y[out])
                if abs(prev_error - curr_error) < 1e-6:
                    if self.printing:
                        print(f"Gradient not changing anymore. Stop at {iter + 1}")
                    break
                prev_error = curr_error

                if self.printing:
                    if iter % 1000 == 0:
                        print(f"Error at [{iter + 1}]: {RMSE(y_train, y[out])}")

                # Backward
                # Note the difference between multiply and dot product
                grad = np.multiply(error, self.deriv_func[out - 1](z[out - 1]))
                for k in range(out - 1, -1, -1):
                    dw[k] = 1 / X_train.shape[0] * np.dot(y[k].T, grad)
                    db[k] = 1 / X_train.shape[0] * np.mean(grad, axis=0)
                    if k > 0:
                        grad = np.multiply(np.dot(grad, w[k].T),
                                           self.deriv_func[k - 1](z[k - 1]))

                for k in range(out):
                    w[k] -= self.learning_rate * dw[k]
                    b[k] -= self.learning_rate * db[k]


            print(f"At running time {running + 1}, RMSE: = {prev_error}")
            if min_error > prev_error:
                min_error = prev_error
                self.w = w
                self.b = b




    def predict(self, X_test):
        y_pred = X_test
        for k in range(len(self.activ_func)):
            z = np.dot(y_pred, self.w[k]) + self.b[k]
            y_pred = self.activ_func[k](z)
        return y_pred
    


    def makePlot(self, X_test, y_pred, ax):
        indices = np.argsort(X_test)

        ax.plot(X_test[indices].reshape(-1),
                y_pred[indices].reshape(-1),
                linewidth=2,
                color="#72E1D1")




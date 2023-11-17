import numpy as np



class SupervisedModel:

    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def _predict(self, row):
        pass

    def makePlot(self, X_test, y_test, y_pred, x, y, ax):
        min_X = X_test[:, x].min()
        max_X = X_test[:, x].max()

        X_graph = np.arange(min_X, max_X, 0.05)
        y_graph = self.predict(X_graph.reshape(-1, 1))

        ax.plot(X_graph, y_graph, linewidth=2, color="#291D25")



class UnsupervisedModel:

    def __init__(self):
        pass

    


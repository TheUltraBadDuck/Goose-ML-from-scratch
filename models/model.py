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

    def makePlot(self, X_test, ax, title, feature):
        min_X = np.min(X_test[:, feature])
        max_X = np.max(X_test[:, feature])

        X_graph = np.arange(min_X, max_X, 0.05)
        y_graph = self.predict(np.reshape(X_graph, (-1, 1)))

        ax.plot(X_graph, y_graph, linewidth=2, color="#291D25")



class UnsupervisedModel:

    def __init__(self):
        pass

    


import numpy as np

from models.model import SupervisedModel



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class NaiveBayes(SupervisedModel):

    def __init__(self):
        pass



    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.y_unique = np.unique(y_train)

        group_len = len(self.y_unique)
        self.mean   = np.zeros((group_len, X_train.shape[1]), dtype=float)
        self.var    = np.zeros((group_len, X_train.shape[1]), dtype=float)
        self.priors = np.zeros(group_len, dtype=float)

        for c in self.y_unique:
            X_train_c = X_train[c == y_train]
            self.mean[c, :] = X_train_c.mean(axis=0)
            self.var[c, :]  = X_train_c.var(axis=0)
            self.priors[c]  = X_train_c.shape[0] / X_train.shape[0]




    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return y_pred
    


    def _predict(self, row):
        # Posterior = Likelihood * Prior / Evidence
        #  P(y|x)   =   P(x|y)   * P(y)  /   P(x)
        # P(x) is not necessary
        posteriors = np.zeros(len(self.y_unique), dtype=float)
        for i in range(len(self.y_unique)):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(getPDF(row, self.mean[i, :], self.var[i, :])))
            posteriors[i] = prior + likelihood
        
        return self.y_unique[np.argmax(posteriors)]

    


def getPDF(row, mean, var):
    num = np.exp(-(row - mean) ** 2 / (2 * var))
    den = np.sqrt(2 * np.pi * var)
    return num / den




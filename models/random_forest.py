import numpy as np
from collections import Counter

from models.model import SupervisedModel
from models.decision_tree import DecisionTree



class RandomForest(SupervisedModel):

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
    


    def fit(self, X_train, y_train):
        self.trees = np.empty(self.n_trees)
        for i in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = self.bootstrap(X_train, y_train)
            tree.fit(X_sample, y_sample)
            self.trees[i] = tree



    def predict(self, X_test):
        tree_preds = np.array([tree.predict(X_test) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self.mostCommonLabel(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)



    def bootstrap(self, X_train, y_train):
        i_list = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
        return X_train[i_list], y_train[i_list]


    
    def mostCommonLabel(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]





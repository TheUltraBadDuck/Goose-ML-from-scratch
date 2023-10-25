import numpy as np
from collections import Counter

from models.model import SupervisedModel
    


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature  = feature
        self.threshold = threshold
        self.left     = left
        self.right    = right
        self.value    = value




class DecisionTree(SupervisedModel):

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
    


    def fit(self, X_train, y_train):
        self.n_feats = X_train.shape[1] if not self.n_feats else min(self.n_feats, X_train.shape[1])
        self.root = self.growTree(X_train, y_train)



    def predict(self, X_test):
        return np.array([self.traverseTree(x, self.root) for x in X_test])



    def makePlot(self, X_test, ax):
        pass


    
    def growTree(self, X_train, y_train, depth=0):

        # Stop
        if (depth >= self.max_depth) or              \
                (np.unique(y_train) == 1).all() or   \
                (X_train.shape[0] < self.min_samples_split):
            leaf = self.mostCommonLabel(y_train)
            return Node(value = leaf)

        feat_i_list = np.random.choice(X_train.shape[1], self.n_feats, replace=False)
        
        best_feat, best_thresh = self.bestCriteria(X_train, y_train, feat_i_list)
        left_i_list  = np.argwhere(X_train[:, best_feat] <= best_thresh).flatten()
        right_i_list = np.argwhere(X_train[:, best_feat] >  best_thresh).flatten()

        left  = self.growTree(X_train[left_i_list, :],  y_train[left_i_list],  depth + 1)
        right = self.growTree(X_train[right_i_list, :], y_train[right_i_list], depth + 1)
        return Node(best_feat, best_thresh, left, right)



    def bestCriteria(self, X_train, y_train, feat_i_list):

        best_gain = -1
        split_i, split_threh = None, None

        for feat_i in feat_i_list:
            X_col = X_train[:, feat_i]
            thresholds = np.unique(X_col)
            
            for threshold in thresholds:
                gain = self.informationGain(y_train, X_col, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_i = feat_i
                    split_threh = threshold
        
        return split_i, split_threh



    def informationGain(self, y_train, X_col, split_threh):
        parent_entropy = self.entropy(y_train)

        left_i_list  = np.argwhere(X_col <= split_threh).flatten()
        right_i_list = np.argwhere(X_col >  split_threh).flatten()

        if (len(left_i_list) == 0) or (len(right_i_list) == 0):
            return 0
        
        en_l = self.entropy(y_train[left_i_list])
        en_r = self.entropy(y_train[right_i_list])

        child_entropy = (left_i_list.shape[0] * en_l + right_i_list.shape[0] * en_r) / y_train.shape[0]

        return parent_entropy - child_entropy



    def mostCommonLabel(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]




    def entropy(self, y):
        unique_list, count_list = np.unique(y, return_counts=True)
        sum = 0.0
        for c in count_list:
            this_sum = c / y.shape[0]
            sum += this_sum * np.log2(this_sum)
        return -sum



    def traverseTree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverseTree(x, node.left)
        return self.traverseTree(x, node.right)



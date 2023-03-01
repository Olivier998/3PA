from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame, Series
import numpy as np


class VariableTree:
    head = None

    def __init__(self, max_depth=4):
        self.dtr = DecisionTreeRegressor(max_depth=max_depth)
        self.features = None

    def fit(self, X, y):
        self.dtr.fit(X, y)
        self.features = X.columns

        self.head = self.add_children(0, X, y)

    def predict(self, X, depth=None):
        if depth is None:
            return self.dtr.predict(X)

        def node_predict(depth):
            def _node_predict(X):
                return self.head.predict(X, depth)
            return _node_predict

        predictions = np.array(X.apply(node_predict(depth), axis=1))  # np.array([self.head.predict(xi) for xi in X])
        return predictions

    def add_children(self, node_id, X, y):
        left_child = self.dtr.tree_.children_left[node_id]
        right_child = self.dtr.tree_.children_right[node_id]

        node_value = y.mean()

        # If we are at a leaf
        if left_child == -1:
            curr_node = _Node(value=node_value)
            return curr_node

        node_thresh = self.dtr.tree_.threshold[node_id]
        node_feature_id = self.dtr.tree_.feature[node_id]
        node_feature = self.features[node_feature_id]

        curr_node = _Node(value=node_value,
                          threshold=node_thresh,
                          feature=node_feature,
                          feature_id=node_feature_id)

        curr_node.c_left = self.add_children(left_child,
                                             X=X.loc[X[node_feature] <= node_thresh],
                                             y=y[X[node_feature] <= node_thresh])
        curr_node.c_right = self.add_children(right_child,
                                              X=X.loc[X[node_feature] > node_thresh],
                                              y=y[X[node_feature] > node_thresh])

        return curr_node


class _Node:
    c_left = None
    c_right = None

    def __init__(self, value, threshold=None, feature=None, feature_id=None):
        self.value = value
        self.threshold = threshold
        self.feature = feature
        self.feature_id = feature_id

    def predict(self, X, depth):
        if depth == 0 or self.c_left is None:
            return self.value

        if type(X) == DataFrame:
            X_value = X[self.feature]  # .iloc[self.feature_id]
        elif type(X) == Series:
            X_value = X[self.feature_id]
        else:
            raise TypeError(f"Parameter X is of type {type(X)}, but it must be of type "
                            f"'pandas.DataFrame' or 'pandas.Series'.")

        if X_value <= self.threshold:  # If node split condition is true, then left children
            return self.c_left.predict(X, depth - 1)
        return self.c_right.predict(X, depth - 1)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    y = np.array([1, 2, 3])
    x = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
    tree = VariableTree()
    tree.fit(x, y)

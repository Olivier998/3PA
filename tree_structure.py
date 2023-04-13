from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame, Series
import numpy as np


class VariableTree:
    head = None

    def __init__(self, max_depth=None, min_sample_ratio=1):
        if min_sample_ratio == 0:
            min_sample_ratio = 1
        else:
            min_sample_ratio = min_sample_ratio / 100
        self.dtr = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_sample_ratio, random_state=54288)
        self.features = None
        self.nb_nodes = 0

    def fit(self, X, y):
        self.dtr.fit(X, y)
        self.features = X.columns

        self.head = self.add_children(0, X, y)

    def get_all_profiles(self, min_ca=0, min_samples_ratio=0):
        profiles = self.head.get_profile(min_samples_ratio=min_samples_ratio, min_ca=min_ca, previous_thresh="*")
        return profiles

    def predict(self, X, depth=None, min_samples_ratio=0):
        if depth is None and min_samples_ratio == 0:
            return self.dtr.predict(X)

        def node_predict(_depth, _min_samples_ratio):
            def _node_predict(X):
                return self.head.predict(X, _depth, _min_samples_ratio)

            return _node_predict

        predictions = np.array(X.apply(node_predict(depth, min_samples_ratio), axis=1))
        return predictions

    def add_children(self, node_id, X, y):
        self.nb_nodes += 1

        left_child = self.dtr.tree_.children_left[node_id]
        right_child = self.dtr.tree_.children_right[node_id]

        node_value = y.mean()

        node_samples_ratio = self.dtr.tree_.n_node_samples[node_id] / self.dtr.tree_.n_node_samples[0] * 100

        # If we are at a leaf
        if left_child == -1:
            curr_node = _Node(value=node_value, samples_ratio=node_samples_ratio, node_id=self.nb_nodes)
            return curr_node

        node_thresh = self.dtr.tree_.threshold[node_id]
        node_feature_id = self.dtr.tree_.feature[node_id]
        node_feature = self.features[node_feature_id]

        curr_node = _Node(value=node_value,
                          samples_ratio=node_samples_ratio,
                          threshold=node_thresh,
                          feature=node_feature,
                          feature_id=node_feature_id,
                          node_id=self.nb_nodes)

        curr_node.c_left = self.add_children(left_child,
                                             X=X.loc[X[node_feature] <= node_thresh],
                                             y=y[X[node_feature] <= node_thresh])
        curr_node.c_right = self.add_children(right_child,
                                              X=X.loc[X[node_feature] > node_thresh],
                                              y=y[X[node_feature] > node_thresh])

        return curr_node

    @property
    def max_depth(self):
        return self.dtr.tree_.max_depth


class _Node:
    c_left = None
    c_right = None

    def __init__(self, value, samples_ratio, threshold=None, feature=None, feature_id=None, node_id=0):
        self.value = value
        self.samples_ratio = samples_ratio
        self.threshold = threshold
        self.feature = feature
        self.feature_id = feature_id
        self.node_id = node_id

    def get_profile(self, min_samples_ratio, min_ca, previous_thresh=""):
        curr_profile_child = []
        prev_thresh_separator = " / " if previous_thresh != "" else ""
        if self.c_left is not None:
            if self.c_left.samples_ratio >= min_samples_ratio:  # self.c_left.value >= min_ca and
                left_prev_thresh = previous_thresh + f"{prev_thresh_separator}{self.feature}<=" \
                                                     f"{round(self.threshold, 2)}"
                curr_profile_child += self.c_left.get_profile(min_samples_ratio=min_samples_ratio,
                                                              min_ca=min_ca, previous_thresh=left_prev_thresh)

        if self.c_right is not None:
            if self.c_right.samples_ratio >= min_samples_ratio:  # self.c_right.value >= min_ca and
                right_prev_thresh = previous_thresh + f"{prev_thresh_separator}{self.feature}>" \
                                                      f"{round(self.threshold, 2)}"
                curr_profile_child += self.c_right.get_profile(min_samples_ratio=min_samples_ratio,
                                                               min_ca=min_ca, previous_thresh=right_prev_thresh)
        if self.value < min_ca and len(curr_profile_child) == 0:
            return []

        return [*curr_profile_child, previous_thresh]

    def predict(self, X, depth=None, min_samples_ratio=0):
        if depth == 0 or self.c_left is None:
            return self.value

        if type(X) == DataFrame:
            X_value = X[self.feature]  # .iloc[self.feature_id]
        elif type(X) == Series:
            X_value = X[self.feature_id]
        else:
            raise TypeError(f"Parameter X is of type {type(X)}, but it must be of type "
                            f"'pandas.DataFrame' or 'pandas.Series'.")

        if depth is not None:
            depth -= 1

        if X_value <= self.threshold:  # If node split condition is true, then left children
            c_node = self.c_left
        else:
            c_node = self.c_right

        if c_node.samples_ratio < min_samples_ratio:  # If not enough samples in child node
            return self.value

        return c_node.predict(X, depth, min_samples_ratio)


if __name__ == '__main__':
    y = np.array([1, 2, 3])
    x = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
    tree = VariableTree()
    tree.fit(x, y)

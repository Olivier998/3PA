from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from pandas import DataFrame, Series
import numpy as np


class VariableTree:
    head = None

    def __init__(self, max_depth=None, min_sample_ratio=1):
        if min_sample_ratio <= 0:
            min_sample_ratio = 1
        else:
            min_sample_ratio = min_sample_ratio / 100
        self.__dtr = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_sample_ratio, random_state=54288)
        self.features = None
        self.nb_nodes = 0

    def fit(self, X, y):
        self.__dtr.fit(X, y)
        self.features = X.columns

        self.head = self.add_children(0, X, y)

    def re_fit(self, X, y):
        if self.head is None:  # If the model if not already fitted
            self.fit(X, y)

        else:
            n0 = X.shape[0]
            self.re_fit_children(self.head, X, y, n0=n0)

    def get_all_profiles(self, min_ca=0, min_samples_ratio=0):
        profiles, nodes_numbers = self.head.get_profile(min_samples_ratio=min_samples_ratio, min_ca=min_ca,
                                                        previous_thresh="*")
        return profiles, nodes_numbers

    def get_node_value(self, node_id, param='value'):
        wanted_node = self.__search_node(node_id=node_id, curr_node=self.head)
        return wanted_node.__getattribute__(param)

    def __search_node(self, node_id, curr_node):
        if curr_node.node_id == node_id:
            return curr_node
        elif curr_node.c_right.node_id > node_id:
            return self.__search_node(node_id=node_id, curr_node=curr_node.c_left)
        return self.__search_node(node_id=node_id, curr_node=curr_node.c_right)

    def predict(self, X, depth=None, min_samples_ratio=0):
        """

        :param X:
        :param depth:
        :param min_samples_ratio:
        :return:
        """

        def node_predict(_depth, _min_samples_ratio):
            def _node_predict(X):
                return self.head.predict(X, _depth, _min_samples_ratio)

            return _node_predict

        predictions = np.array(X.apply(node_predict(depth, min_samples_ratio), axis=1))
        return predictions

    def add_children(self, node_id, X, y):
        self.nb_nodes += 1

        left_child = self.__dtr.tree_.children_left[node_id]
        right_child = self.__dtr.tree_.children_right[node_id]

        node_value = y.mean()
        node_max = y.max()

        node_samples_ratio = self.__dtr.tree_.n_node_samples[node_id] / self.__dtr.tree_.n_node_samples[0] * 100

        # If we are at a leaf
        if left_child == -1:
            curr_node = _Node(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio,
                              node_id=self.nb_nodes)
            return curr_node

        node_thresh = self.__dtr.tree_.threshold[node_id]
        node_feature_id = self.__dtr.tree_.feature[node_id]
        node_feature = self.features[node_feature_id]

        curr_node = _Node(value=node_value,
                          value_max=node_max,
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

    def re_fit_children(self, curr_node, X, y, n0):
        node_value = y.mean() if len(y)>0 else 0
        node_max = y.max() if len(y)>0 else 0
        node_samples_ratio = X.shape[0] / n0 * 100

        curr_node.re_fit(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio)

        if curr_node.c_left is not None:
            node_feature = curr_node.feature
            node_thresh = curr_node.threshold
            self.re_fit_children(curr_node.c_left,
                                 X=X.loc[X[node_feature] <= node_thresh],
                                 y=y[X[node_feature] <= node_thresh],
                                 n0=n0)
            self.re_fit_children(curr_node.c_right,
                                 X=X.loc[X[node_feature] > node_thresh],
                                 y=y[X[node_feature] > node_thresh],
                                 n0=n0)

    @property
    def max_depth(self):
        return self.__dtr.tree_.max_depth


class _Node:
    c_left = None
    c_right = None

    def __init__(self, value, value_max, samples_ratio, threshold=None, feature=None, feature_id=None, node_id=0):
        self.value = value
        self.value_max = value_max
        self.samples_ratio = samples_ratio
        self.threshold = threshold
        self.feature = feature
        self.feature_id = feature_id
        self.node_id = node_id

    def re_fit(self, value, value_max, samples_ratio):
        self.value = value
        self.value_max = value_max
        self.samples_ratio = samples_ratio

    def get_profile(self, min_samples_ratio, min_ca, previous_thresh=""):
        curr_profile_child = []
        curr_child_nodeid = []
        prev_thresh_separator = " / " if previous_thresh != "" else ""
        temp = 0
        if self.c_left is not None:
            if self.c_left.samples_ratio >= min_samples_ratio:  # self.c_left.value >= min_ca and
                temp += 1
                left_prev_thresh = previous_thresh + f"{prev_thresh_separator}{self.feature}<=" \
                                                     f"{round(self.threshold, 2)}"
                c_prev_str, c_prev_id = self.c_left.get_profile(min_samples_ratio=min_samples_ratio,
                                                                min_ca=min_ca, previous_thresh=left_prev_thresh)
                curr_profile_child += c_prev_str
                curr_child_nodeid += c_prev_id

        if self.c_right is not None:
            if self.c_right.samples_ratio >= min_samples_ratio:  # self.c_right.value >= min_ca and
                temp += 1
                right_prev_thresh = previous_thresh + f"{prev_thresh_separator}{self.feature}>" \
                                                      f"{round(self.threshold, 2)}"
                c_prev_str, c_prev_id = self.c_right.get_profile(min_samples_ratio=min_samples_ratio,
                                                               min_ca=min_ca, previous_thresh=right_prev_thresh)
                curr_profile_child += c_prev_str
                curr_child_nodeid += c_prev_id

        if min_samples_ratio < 0:  # Case where we don't use profiles
            if self.value_max < min_ca and len(curr_profile_child) == 0:
                return [], []
        else:
            if self.value < min_ca and len(curr_profile_child) == 0:
                return [], []

        return [*curr_profile_child, previous_thresh], [*curr_child_nodeid, self.node_id]

    def predict(self, X, depth=None, min_samples_ratio=0):
        if depth == 0 or self.c_left is None:
            return self.value

        if type(X) == DataFrame:
            X_value = X[self.feature]
        elif type(X) == Series:
            X_value = X.iloc[self.feature_id]
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

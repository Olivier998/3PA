from bokeh.models import Rect, Arrow
from tree_structure import VariableTree, _Node
from utils import get_mdr
import numpy as np


class TreeTranscriber:

    def __init__(self, tree: VariableTree, dimensions=[20, 10], min_ratio_leafs: float = 0.5, THRESHOLD=0.5):
        self.tree = tree
        self.width = dimensions[0]
        self.height = dimensions[1]
        self.min_ratio_leafs = min_ratio_leafs
        self.THRESHOLD = THRESHOLD

    def render_to_bokeh(self, x, y_true, y_prob, min_cas, depth=None, *kwargs):
        if depth is None:
            depth = self.tree.max_depth

        nodes, arrows = self.__add_node(x=x, y_true=y_true, y_prob=y_prob, min_cas=min_cas, curr_node=self.tree.head,
                                        remaining_depth=depth, curr_depth=0, *kwargs)

        return nodes, arrows

    def __add_node(self, x, y_true, y_prob, min_cas, remaining_depth, curr_node, curr_depth, pos_x=0, pos_y=0, *kwargs):
        values_depth_dr = {'depth': [], 'values': []}
        y_pred = np.array([1 if y_score_i >= self.THRESHOLD else 0 for y_score_i in y_prob])
        for depth in range(curr_depth, self.tree.max_depth + 1):
            mdr_values = get_mdr(Y_target=y_true, Y_predicted=y_pred, predicted_accuracies=min_cas[depth])
            mdr_dict = {'dr': [dic['dr'] for dic in mdr_values]}
            mdr_dict['metrics'] = [{key: value for key, value in dic.items() if key != 'dr'} for dic in mdr_values]

            values_depth_dr['depth'].append(depth)
            values_depth_dr['values'].append(mdr_dict)

        rect = Rect(x=pos_x, y=pos_y, width=self.width, height=self.height, fill_color='white', line_color='black',
                    line_width=2, tags=[curr_depth, values_depth_dr])

        if remaining_depth == 0 or curr_node.c_left is None:
            return [rect], []

        # left child nodes
        pos_x_left = pos_x - 1 / 2 * self.width * (1 + self.min_ratio_leafs) * 2 ** (remaining_depth - 1)
        pos_y_left = pos_y - 2 * self.height

        idx_left = [row_idx for row_idx, row_val in x.iterrows() if row_val[curr_node.feature] <= curr_node.threshold]
        min_cas_left = {}
        for depth in min_cas:
            min_cas_left[depth] = min_cas[depth][idx_left]

        c_left = self.__add_node(x=x.iloc[idx_left],
                                 y_true=y_true[idx_left],
                                 y_prob=y_prob[idx_left],
                                 min_cas=min_cas_left,
                                 pos_x=pos_x_left,
                                 pos_y=pos_y_left,
                                 curr_node=curr_node.c_left,
                                 remaining_depth=remaining_depth - 1,
                                 curr_depth=curr_depth + 1)

        arr_left = Arrow(x_start=pos_x, y_start=pos_y - 0.5 * self.height, x_end=pos_x_left,
                         y_end=pos_y_left + 0.5 * self.height, end=None, line_color="green", line_width=3)

        # right child nodes
        pos_x_right = pos_x + 1 / 2 * self.width * (1 + self.min_ratio_leafs) * 2 ** (remaining_depth - 1)
        pos_y_right = pos_y - 2 * self.height

        idx_right = [row_idx for row_idx, row_val in x.iterrows() if row_val[curr_node.feature] > curr_node.threshold]
        min_cas_right = {}
        for depth in min_cas:
            min_cas_right[depth] = min_cas[depth][idx_right]

        c_right = self.__add_node(x=x.iloc[idx_right],
                                  y_true=y_true[idx_right],
                                  y_prob=y_prob[idx_right],
                                  min_cas=min_cas_right,
                                  pos_x=pos_x_right,
                                  pos_y=pos_y_right,
                                  curr_node=curr_node.c_right,
                                  remaining_depth=remaining_depth - 1,
                                  curr_depth=curr_depth + 1)
        arr_right = Arrow(x_start=pos_x, y_start=pos_y - 0.5 * self.height, x_end=pos_x_right,
                          y_end=pos_y_right + 0.5 * self.height, end=None, line_color="red", line_width=3)

        return [rect, *c_left[0], *c_right[0]], [arr_left, arr_right, *c_left[1], *c_right[1]]

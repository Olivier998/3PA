from bokeh.models import Rect, Arrow
from tree_structure import VariableTree, _Node
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score


class TreeTranscriber:

    def __init__(self, tree: VariableTree, dimensions=[20, 16], min_ratio_leafs: float = 0.5, THRESHOLD=0.5,
                 metrics=None):
        self.tree = tree
        self.width = dimensions[0]
        self.height = dimensions[1]
        self.min_ratio_leafs = min_ratio_leafs
        self.THRESHOLD = THRESHOLD
        self.metrics = metrics if metrics is not None else ['bal_acc']

    def render_to_bokeh(self, x, y_true, y_prob, min_cas, depth=None, **kwargs):
        if depth is None:
            depth = self.tree.max_depth

        mdr_tool = MDR(pred_cas=min_cas)

        nodes, arrows, text = self.__add_node(x=x, y_true=y_true, y_prob=y_prob, min_cas=min_cas,
                                              curr_node=self.tree.head, mdr_tool=mdr_tool,
                                              remaining_depth=depth, curr_depth=0, **kwargs)

        return nodes, arrows, text

    def __add_node(self, x, y_true, y_prob, min_cas, mdr_tool, remaining_depth, curr_node, curr_depth, pos_x=0, pos_y=0,
                   **kwargs):
        values_depth_dr = {'depth': [], 'values': []}
        y_pred = np.array([1 if y_score_i >= self.THRESHOLD else 0 for y_score_i in y_prob])
        for depth in range(curr_depth, self.tree.max_depth + 1):
            mdr_values = mdr_tool.get_metrics(Y_target=y_true, Y_predicted=y_pred, pred_cas=min_cas[depth], depth=depth)
            mdr_dict = {'dr': [dic['dr'] for dic in mdr_values]}
            mdr_dict['metrics'] = [{key: value for key, value in dic.items() if key != 'dr'} for dic in mdr_values]

            values_depth_dr['depth'].append(depth)
            values_depth_dr['values'].append(mdr_dict)

        rect = Rect(x=pos_x, y=pos_y, width=self.width, height=self.height, fill_color='white', line_color='black',
                    line_width=2, tags=[{'curr_depth': curr_depth, 'node_id': curr_node.node_id},
                                        values_depth_dr])

        node_text = [{'x': pos_x - 0.45 * self.width, 'y': pos_y+(0.375 - 0.125 * idx) * self.height, 'text': f'{idx}',
                      'metric': metric_name, 'node_id': curr_node.node_id, 'curr_depth': curr_depth}
                     for idx, metric_name in enumerate(self.metrics)]

        if remaining_depth == 0 or curr_node.c_left is None:
            return [rect], [], node_text

        # left child nodes
        pos_x_left = pos_x - 1 / 2 * self.width * (1 + self.min_ratio_leafs) * 2 ** (remaining_depth - 1)
        pos_y_left = pos_y - 2 * self.height

        idx_left = [row_idx for row_idx, row_val in x.iterrows() if row_val[curr_node.feature] <= curr_node.threshold]
        min_cas_left = {}
        for depth in min_cas:
            min_cas_left[depth] = min_cas[depth][idx_left]

        c_left = self.__add_node(x=x.iloc[idx_left].reset_index(drop=True),
                                 y_true=y_true[idx_left],
                                 y_prob=y_prob[idx_left],
                                 min_cas=min_cas_left,
                                 mdr_tool=mdr_tool,
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

        c_right = self.__add_node(x=x.iloc[idx_right].reset_index(drop=True),
                                  y_true=y_true[idx_right],
                                  y_prob=y_prob[idx_right],
                                  min_cas=min_cas_right,
                                  mdr_tool=mdr_tool,
                                  pos_x=pos_x_right,
                                  pos_y=pos_y_right,
                                  curr_node=curr_node.c_right,
                                  remaining_depth=remaining_depth - 1,
                                  curr_depth=curr_depth + 1)
        arr_right = Arrow(x_start=pos_x, y_start=pos_y - 0.5 * self.height, x_end=pos_x_right,
                          y_end=pos_y_right + 0.5 * self.height, end=None, line_color="red", line_width=3)

        return [rect, *c_left[0], *c_right[0]], \
               [arr_left, arr_right, *c_left[1], *c_right[1]], \
               [*node_text, *c_left[2], *c_right[2]]


class MDR:
    def __init__(self, pred_cas, precision=2):
        self.n_total = {depth: len(pred_cas[depth]) for depth in pred_cas}
        unique_accuracies = {depth: np.sort(np.unique(pred_cas[depth]))[::-1] for depth in pred_cas}
        self.dr = {depth: {min_perc: sum(pred_cas[depth] >= min_perc) / self.n_total[depth]
                           for min_perc in unique_accuracies[depth]} for depth in pred_cas}
        self.precision = precision

    def get_metrics(self, Y_target, Y_predicted, pred_cas, depth):
        unique_accuracies = np.sort(np.unique(pred_cas))[::-1]
        mdr_values = []

        for min_perc in unique_accuracies:
            dr = self.dr[depth][min_perc]
            perc_node = sum(pred_cas >= min_perc) / len(Y_target)
            perc_pop = sum(pred_cas >= min_perc) / self.n_total[depth]
            acc = accuracy_score(Y_target[pred_cas >= min_perc],
                                 Y_predicted[pred_cas >= min_perc])
            auc = roc_auc_score(Y_target[pred_cas >= min_perc],
                                Y_predicted[pred_cas >= min_perc]) if \
                len(np.unique(Y_target[pred_cas >= min_perc])) > 1 else 0
            bal_acc = balanced_accuracy_score(Y_target[pred_cas >= min_perc],
                                              Y_predicted[pred_cas >= min_perc])
            sensitivity = recall_score(Y_target[pred_cas >= min_perc],
                                       Y_predicted[pred_cas >= min_perc]
                                       , pos_label=1, zero_division=0)
            specificity = recall_score(Y_target[pred_cas >= min_perc],
                                       Y_predicted[pred_cas >= min_perc]
                                       , pos_label=0, zero_division=0)
            mean_ca = np.mean(pred_cas[pred_cas >= min_perc])
            mdr_values.append({'dr': dr, 'accuracy': acc, 'bal_acc': bal_acc,
                               'sens': sensitivity, 'spec': specificity, 'perc_node': perc_node,
                               'perc_pop': perc_pop, 'auc': auc, 'mean_ca': mean_ca})

        for i, values in enumerate(mdr_values):
            for metric in values:
                mdr_values[i][metric] = round(values[metric], self.precision)
        mdr_values = np.array(mdr_values)

        return mdr_values

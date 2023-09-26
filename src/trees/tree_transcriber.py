from bokeh.models import Rect, Arrow
from src.trees.tree_structure import VariableTree
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score, matthews_corrcoef, \
    f1_score
from settings.layout_parameters import ITALIC_VARS

# Remove scikit warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


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
        values_sampratio_dr = {'samp_ratio': [], 'values': []}
        y_pred = np.array([1 if y_score_i >= self.THRESHOLD else 0 for y_score_i in y_prob])

        for samp_ratio in min_cas:
            mdr_values = mdr_tool.get_metrics(Y_target=y_true, Y_predicted=y_pred, Y_prob=y_prob,
                                              pred_cas=min_cas[samp_ratio], samp_ratio=samp_ratio)
            mdr_dict = {'dr': [dic['dr'] for dic in mdr_values],
                        'metrics': [{key: value for key, value in dic.items() if key != 'dr'} for dic in mdr_values]}
            values_sampratio_dr['samp_ratio'].append(samp_ratio)
            values_sampratio_dr['values'].append(mdr_dict)

        if remaining_depth == 0 or (curr_node.c_left is None and curr_node.c_right is None):
            node_split = ''
        else:
            node_split = f'{curr_node.feature} <= {round(curr_node.threshold, 4)}'

        rect = Rect(x=pos_x, y=pos_y, width=self.width, height=self.height, fill_color='white', line_color='black',
                    line_width=2, tags=[{'curr_depth': curr_depth, 'node_id': curr_node.node_id,
                                         'split': node_split,
                                         'samp_ratio': curr_node.samples_ratio},
                                        values_sampratio_dr])

        node_text = [{'x': pos_x - 0.45 * self.width, 'y': pos_y + (0.375 - 0.12 * idx) * self.height, 'text': '',
                      'metric': metric_name, 'text_font_style': 'italic' if metric_name in ITALIC_VARS else 'normal',
                      'node_id': curr_node.node_id, 'curr_depth': curr_depth}
                     for idx, metric_name in enumerate(self.metrics)]

        if remaining_depth == 0 or (curr_node.c_left is None and curr_node.c_right is None):
            return [rect], [], node_text

        # Add split criteria
        node_text.append({'x': pos_x - 0.45 * self.width, 'y': pos_y - 0.75 * self.height,
                          'text': f'{curr_node.feature} <= {curr_node.threshold}',
                          'text_font_style': 'normal',
                          'metric': 'split', 'node_id': curr_node.node_id, 'curr_depth': curr_depth})

        # Child lists
        c_rect = [rect]
        c_arr = []
        c_text = node_text

        # left child nodes
        if curr_node.c_left is not None:
            pos_x_left = pos_x - 1 / 2 * self.width * (1 + self.min_ratio_leafs) * 2 ** (remaining_depth - 1)
            pos_y_left = pos_y - 2 * self.height

            idx_left = [row_idx for row_idx, row_val in x.iterrows() if
                        row_val[curr_node.feature] <= curr_node.threshold]
            min_cas_left = {}
            for samp_ratio in min_cas:
                min_cas_left[samp_ratio] = min_cas[samp_ratio][idx_left]

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
            c_rect += c_left[0]
            c_arr += c_left[1]
            c_text += c_left[2]

            c_arr += [Arrow(x_start=pos_x, y_start=pos_y - 0.5 * self.height, x_end=pos_x_left,
                            y_end=pos_y_left + 0.5 * self.height, end=None, line_color="green", line_width=3,
                            tags=[{'node_id': curr_node.c_left.node_id}])]

        # right child nodes
        if curr_node.c_right is not None:
            pos_x_right = pos_x + 1 / 2 * self.width * (1 + self.min_ratio_leafs) * 2 ** (remaining_depth - 1)
            pos_y_right = pos_y - 2 * self.height

            idx_right = [row_idx for row_idx, row_val in x.iterrows() if
                         row_val[curr_node.feature] > curr_node.threshold]
            min_cas_right = {}
            for samp_ratio in min_cas:
                min_cas_right[samp_ratio] = min_cas[samp_ratio][idx_right]

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
            c_rect += c_right[0]
            c_arr += c_right[1]
            c_text += c_right[2]

            c_arr += [Arrow(x_start=pos_x, y_start=pos_y - 0.5 * self.height, x_end=pos_x_right,
                            y_end=pos_y_right + 0.5 * self.height, end=None, line_color="red", line_width=3,
                            tags=[{'node_id': curr_node.c_right.node_id}])]

        return c_rect, c_arr, c_text


class MDR:
    def __init__(self, pred_cas, precision=2):
        self.n_total = {samp_ratio: len(pred_cas[samp_ratio]) for samp_ratio in pred_cas}
        # unique_accuracies = {samp_ratio: np.sort(np.unique(pred_cas[samp_ratio]))[::-1] for samp_ratio in pred_cas}
        # self.dr = {samp_ratio: {min_perc: sum(pred_cas[samp_ratio] >= min_perc) / self.n_total[samp_ratio]
        #                        for min_perc in unique_accuracies[samp_ratio]} for samp_ratio in pred_cas}

        # Get DR only if min_ca is different
        self.dr = {}
        for samp_ratio in pred_cas:
            self.dr[samp_ratio] = {}
            prev_min_acc = -1
            for dr in range(100, 0, -1):
                curr_min_acc = np.sort(pred_cas[samp_ratio])[int(len(pred_cas[samp_ratio]) * (1 - dr / 100))]
                if prev_min_acc < curr_min_acc:
                    prev_min_acc = curr_min_acc
                    self.dr[samp_ratio][dr] = curr_min_acc

        # self.dr = {samp_ratio: {dr: np.sort(pred_cas[samp_ratio])[int(len(pred_cas[samp_ratio]) * (1-dr/100))] for
        #                        dr in range(100, 0, -1)} for samp_ratio in pred_cas}
        self.precision = precision

    def get_metrics(self, Y_target, Y_predicted, Y_prob, pred_cas, samp_ratio):
        # unique_accuracies = np.sort(np.unique(np.round(pred_cas, 3)))[::-1]
        # sorted_accuracies = np.sort(pred_cas)[::-1]
        mdr_values = []

        for dr, dr_accuracy in self.dr[samp_ratio].items():
            # dr_accuracy = self.dr[samp_ratio][dr]  # np.percentile(pred_cas, 100 - dr, interpolation="lower")
            if sum(pred_cas >= dr_accuracy) > 0:
                perc_node = sum(pred_cas >= dr_accuracy) * 100 / len(Y_target)
                perc_pop = sum(pred_cas >= dr_accuracy) * 100 / self.n_total[samp_ratio]
                acc = accuracy_score(Y_target[pred_cas >= dr_accuracy],
                                     Y_predicted[pred_cas >= dr_accuracy]) * 100
                auc = roc_auc_score(Y_target[pred_cas >= dr_accuracy],
                                    Y_prob[pred_cas >= dr_accuracy]) * 100 if \
                    len(np.unique(Y_target[pred_cas >= dr_accuracy])) > 1 else 0
                auprc = average_precision_score(Y_target[pred_cas >= dr_accuracy],
                                                Y_prob[pred_cas >= dr_accuracy]) * 100 if \
                    len(np.unique(Y_target[pred_cas >= dr_accuracy])) > 1 else 0
                mcc = matthews_corrcoef(Y_target[pred_cas >= dr_accuracy],
                                        Y_predicted[pred_cas >= dr_accuracy]) * 100 if \
                    len(np.unique(Y_target[pred_cas >= dr_accuracy])) > 1 else 0


                f1score = f1_score(Y_target[pred_cas >= dr_accuracy],
                                   Y_predicted[pred_cas >= dr_accuracy],
                                   zero_division=0) * 100 if \
                    len(np.unique(Y_target[pred_cas >= dr_accuracy])) > 1 else 0

                # bal_acc = balanced_accuracy_score(Y_target[pred_cas > dr_accuracy],
                #                                  Y_predicted[pred_cas > dr_accuracy])
                sensitivity = recall_score(Y_target[pred_cas >= dr_accuracy],
                                           Y_predicted[pred_cas >= dr_accuracy]
                                           , pos_label=1, zero_division=0) * 100
                specificity = recall_score(Y_target[pred_cas >= dr_accuracy],
                                           Y_predicted[pred_cas >= dr_accuracy]
                                           , pos_label=0, zero_division=0) * 100
                bal_acc = (sensitivity + specificity) / 2
                mean_ca = np.mean(pred_cas[pred_cas >= dr_accuracy]) * 100 if \
                    pred_cas[pred_cas >= dr_accuracy].size > 0 \
                    else np.NaN
                pos_class_occurence = np.sum(Y_target[pred_cas >= dr_accuracy]) / \
                                      len(Y_target[pred_cas >= dr_accuracy]) * 100
                mdr_values.append({'dr': dr / 100, 'accuracy': acc, 'bal_acc': bal_acc,
                                   'sens': sensitivity, 'spec': specificity, 'perc_node': perc_node,
                                   'perc_pop': perc_pop, 'auc': auc, 'auprc': auprc, 'mean_ca': mean_ca,
                                   'pos_perc': pos_class_occurence, 'mcc': mcc, 'f1score': f1score})

        for i, values in enumerate(mdr_values):
            for metric in values:
                mdr_values[i][metric] = round(values[metric], self.precision)
        mdr_values = np.array(mdr_values)
        return mdr_values

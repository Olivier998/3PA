from io import StringIO
import numpy as np
from numbers import Integral
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, roc_curve


class Sentinel:
    def __repr__(self):
        return '"tree.dot"'


SENTINEL = Sentinel()


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class _BaseTreeExporter:
    def __init__(
            self,
            max_depth=None,
            feature_names=None,
            class_names=None,
            label="all",
            filled=False,
            impurity=True,
            node_ids=False,
            proportion=False,
            rounded=False,
            precision=3,
            fontsize=None,
    ):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors["bounds"] is None:
            # Classification tree
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # Regression tree or multi-output
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                    self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion, decision_tree, parent):
        node_values = pd.DataFrame(columns=['tree_train', 'sl_train'])

        # revoir cest quoi original data vs test data, genre c'est quoi train ou test
        # original c'est tree_train ou encore test set


        original_data = decision_tree.original_data
        test_data = decision_tree.test_data
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = '|'  # characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != -1:  # _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            node_string += "%s %s %s%s" % (
                feature,
                characters[3],
                round(tree.threshold[node_id], self.precision),
                characters[4],
            )
            # Features splits
            decision_tree.features_splits[tree.children_left[node_id]] += f"{decision_tree.features_splits[node_id]}" \
                                                                          f"{feature} <= {tree.threshold[node_id]}  "
            decision_tree.features_splits[tree.children_right[node_id]] += f"{decision_tree.features_splits[node_id]}" \
                                                                           f"{feature} > {tree.threshold[node_id]}  "
            # Assign data to childrens
            # Training data
            data_train = original_data[node_id]
            original_data[tree.children_left[node_id]] = data_train[data_train[feature] <= tree.threshold[node_id]]
            original_data[tree.children_right[node_id]] = data_train[data_train[feature] > tree.threshold[node_id]]
            # Testing data
            data_test = test_data[node_id]
            test_data[tree.children_left[node_id]] = data_test[data_test[feature] <= tree.threshold[node_id]]
            test_data[tree.children_right[node_id]] = data_test[data_test[feature] > tree.threshold[node_id]]

        #if parent is not None:
        #    prev_feature = self.feature_names[tree.feature[parent]]
            # Add percentage of missing data of previous decision feature
            #tree_missing_perc = round(original_data[node_id][f'miss_{prev_feature}'].mean() * 100, self.precision)
            #sl_missing_perc = round(test_data[node_id][f'miss_{prev_feature}'].mean() * 100, self.precision)
            #node_values.loc['Missing feature', 'tree_train'] = f"{tree_missing_perc} "
            #node_values.loc['Missing feature', 'sl_train'] = f"{sl_missing_perc} "

        #else:
            # Add percentage of missing data of previous decision feature
            #node_values.loc['Missing feature'] = ''

        # Write impurity
        """if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
                criterion = "squared_error"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += "%s = " % criterion
            node_string += (
                str(round(tree.impurity[node_id], self.precision)) + characters[4]
            )"""
        # Write node sample count
        perc_tree_train = 100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
        node_values.loc['# patients', 'tree_train'] = f"{tree.n_node_samples[node_id]} ({round(perc_tree_train, 1)})"

        perc_sl_train = 100.0 * test_data[node_id].shape[0] / test_data[0].shape[0]
        node_values.loc['# patients', 'sl_train'] = f"{test_data[node_id].shape[0]} ({round(perc_sl_train, 1)})"

        """if labels:
            node_string += "#1 = "
        if self.proportion:
            percent = (
                    100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            )
            node_string += f"{tree.n_node_samples[node_id]} ({round(percent, 1)}%) {characters[4]}"

            node_string += "#2 = "
            percent_test = (
                    100.0 * test_data[node_id].shape[0] / test_data[0].shape[0]
            )
            node_string += f"{test_data[node_id].shape[0]} ({round(percent_test, 1)}%) {characters[4]}"
            # str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]
            # node_string += "#2 = "
            # node_string += str(tree.n_node_samples[node_id]) + characters[4]
        """
        # Write node class distribution / regression value
        #node_values.loc['misclassified', 'tree_train'] = round(original_data[node_id]['misclassified'].mean(), self.precision)
        #node_values.loc['misclassified', 'sl_train'] = round(test_data[node_id]['misclassified'].mean(), self.precision)
        """
        if self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += "P(misclass) = "
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value[1], self.precision)
            # print(f"1{value_text}")
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype("S32")).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")

        node_string += value_text + characters[4]
        """

        # Add mean of predicted probabilities
        tree_pred_prob = np.around(original_data[node_id]['prediction'].mean(), self.precision)
        tree_pred_std = np.around(original_data[node_id]['prediction'].std(), self.precision)
        node_values.loc['prediction', 'tree_train'] = f"{tree_pred_prob} ({tree_pred_std})"

        sl_pred_prob = np.around(test_data[node_id]['prediction'].mean(), self.precision)
        sl_pred_std = np.around(test_data[node_id]['prediction'].std(), self.precision)
        node_values.loc['prediction', 'sl_train'] = f"{sl_pred_prob} ({sl_pred_std})"

        # Add percentage of deceased
        tree_deceased_prob = np.around(original_data[node_id]['deceased'].mean(), self.precision)
        tree_deceased_std = np.around(original_data[node_id]['deceased'].std(), self.precision)
        node_values.loc['deceased', 'tree_train'] = f"{tree_deceased_prob} ({tree_deceased_std})"

        sl_deceased_prob = np.around(test_data[node_id]['deceased'].mean(), self.precision)
        sl_deceased_std = np.around(test_data[node_id]['deceased'].std(), self.precision)
        node_values.loc['deceased', 'sl_train'] = f"{sl_deceased_prob} ({sl_deceased_std})"

        # Add percentage of missing data (global)
        #tree_missing_cols = [col for col in original_data[node_id].columns if 'miss_' in col]
        #sl_missing_cols = [col for col in test_data[node_id].columns if 'miss_' in col]

        #tree_missing_data = np.around(original_data[node_id][tree_missing_cols].mean().mean()*100, self.precision)
        #node_values.loc['Missing data', 'tree_train'] = f"{tree_missing_data}"

        #sl_missing_data = np.around(test_data[node_id][sl_missing_cols].mean().mean()*100, self.precision)
        #node_values.loc['Missing data', 'sl_train'] = f"{sl_missing_data}"

        # Add best threshold
        #ori_fpr, ori_tpr, ori_thresholds = roc_curve(original_data[node_id]['deceased'],
        #                                             original_data[node_id]['prediction'],
        #                                             pos_label=1)
        # calculate the g-mean for each threshold
        #ori_gmeans = np.sqrt(ori_tpr * (1 - ori_fpr))
        #ori_best_thresh = ori_thresholds[np.argmax(ori_gmeans)]
        #node_values.loc['threshold', 'tree_train'] = f"{np.around(ori_best_thresh, self.precision)}"

        # original_data[node_id].loc[:, ['predicted_label']] = original_data[node_id]['prediction'] >= ori_best_thresh

        #test_fpr, test_tpr, test_thresholds = roc_curve(test_data[node_id]['deceased'],
        #                                                test_data[node_id]['prediction'],
        #                                                pos_label=1)
        # calculate the g-mean for each threshold
        #test_gmeans = np.sqrt(test_tpr * (1 - test_fpr))
        #test_best_thresh = test_thresholds[np.argmax(test_gmeans)]
        #node_values.loc['threshold', 'sl_train'] = f"{np.around(test_best_thresh, self.precision)}"

        # test_data[node_id].loc[:, ['predicted_label']] = test_data[node_id]['prediction'] >= test_best_thresh

        # Add sensitivity
        #node_values.loc['sensitivity', 'tree_train'] = \
        #    f"{np.around(recall_score(original_data[node_id]['deceased'],original_data[node_id]['predicted_label'], pos_label=1, zero_division=0), self.precision)}"

        #node_values.loc['sensitivity', 'sl_train'] = \
        #    f"{np.around(recall_score(test_data[node_id]['deceased'],test_data[node_id]['predicted_label'], pos_label=1, zero_division=0), self.precision)}"

        # Add specificity
        #node_values.loc['specificity', 'tree_train'] = \
        #    f"{np.around(recall_score(original_data[node_id]['deceased'], original_data[node_id]['predicted_label'], pos_label=0, zero_division=0), self.precision)}"

        #node_values.loc['specificity', 'sl_train'] = \
        #    f"{np.around(recall_score(test_data[node_id]['deceased'], test_data[node_id]['predicted_label'], pos_label=0, zero_division=0), self.precision)}"

        # Add auc
        #try:
        #    tree_auc = np.around(roc_auc_score(original_data[node_id]['deceased'], original_data[node_id]['predicted_label']), self.precision)
        #except Exception as e:
        #    tree_auc = ""
        #try:
        #    sl_auc = np.around(roc_auc_score(test_data[node_id]['deceased'], test_data[node_id]['predicted_label']), self.precision)
        #except Exception as e:
        #    sl_auc = ""
        #node_values.loc['auc', 'tree_train'] = f"{tree_auc}"

        #node_values.loc['auc', 'sl_train'] = f"{sl_auc}"

        """
        node_string += f"P_hat(C1) = {pred_prob} ({pred_std}) {characters[4]}"

        if self.proportion:
            node_string += "1-P(C1) = "
            # print(f"2{original_data[node_id]['misclassified'].mean()}")
            mean_1 = round(original_data[node_id]['deceased'].mean(), self.precision)
            var = round(original_data[node_id]['deceased'].var(), self.precision)
            value_text = f"{mean_1} ({var})"

        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        if self.proportion:
            probtest = round(test_data[node_id]['deceased'].mean(), self.precision)
            vartest = round(test_data[node_id]['deceased'].var(), self.precision)
            node_string += f"2-P(C1) = {probtest} ({vartest}) {characters[4]}"

            # probmissing1 = round(original_data[node_id]['missing'].sum() / test_data[node_id].shape[0], self.precision)
            # node_string += f"1-P(miss) = {probmissing1} {characters[4]}"
            # probmissing2 = round(test_data[node_id]['missing'].sum() / test_data[node_id].shape[0], self.precision)
            # node_string += f"2-P(miss) = {probmissing2} {characters[4]}"
        """
        # statist = stats.ttest_ind(original_data[node_id]['misclassified'], test_data[node_id]['misclassified'])
        # node_string += f"ttest:{round(statist.statistic, self.precision)} " \
        #               f"pval:{round(statist.pvalue, self.precision)} {characters[4]}"

        # Write node majority class
        if (
                self.class_names is not None
                and tree.n_classes[0] != 1
                and tree.n_outputs == 1
        ):
            # Only done for single-output classification trees
            if labels:
                node_string += "misclassification = "
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(value),
                    characters[2],
                )
            node_string += class_name
            # class_test = "1" if round(test_data[node_id]['misclassified'].mean(), self.precision) >= 0.5 else "0"
            # if class_name != class_test:
            #    node_string += " !!!"

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        return node_values  # node_string + characters[5]


class _DOTTreeExporter(_BaseTreeExporter):
    def __init__(
            self,
            out_file=SENTINEL,
            max_depth=None,
            feature_names=None,
            class_names=None,
            label="all",
            filled=False,
            leaves_parallel=False,
            impurity=True,
            node_ids=False,
            proportion=False,
            rotate=False,
            rounded=False,
            special_characters=False,
            precision=3,
            fontname="helvetica",
    ):

        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters
        self.fontname = fontname
        self.rotate = rotate

        # PostScript compatibility for special characters
        if special_characters:
            self.characters = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
        else:
            self.characters = ["#", "[", "]", "<=", "\\n", '"']

        # validate
        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError(
                    "'precision' should be greater or equal to 0."
                    " Got {} instead.".format(precision)
                )
        else:
            raise ValueError(
                "'precision' should be an integer. Got {} instead.".format(
                    type(precision)
                )
            )

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

    def export(self, decision_tree):
        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_in_ in the decision_tree
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_in_:
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), decision_tree.n_features_in_)
                )
        # each part writes to out_file
        self.head()
        # Now recurse the tree and add node & edge attributes
        # if isinstance(decision_tree, _tree.Tree):
        #    self.recurse(decision_tree, 0, criterion="impurity")
        # else:
        self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion,
                     decision_tree=decision_tree)

        self.tail()

    def tail(self):
        # If required, draw leaf nodes at same depth as each other
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write(
                    "{rank=same ; " + "; ".join(r for r in self.ranks[rank]) + "} ;\n"
                )
        self.out_file.write("}")

    def head(self):
        self.out_file.write("digraph Tree {\n")

        # Specify node aesthetics
        self.out_file.write("node [shape=record")
        rounded_filled = []
        if self.filled:
            rounded_filled.append("filled")
        if self.rounded:
            rounded_filled.append("rounded")
        if len(rounded_filled) > 0:
            self.out_file.write(
                ', style="%s", color="black"' % ", ".join(rounded_filled)
            )

        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write("] ;")

        # Specify graph & edge aesthetics
        if self.leaves_parallel:
            self.out_file.write("graph [ranksep=equally, splines=polyline] ;")

        self.out_file.write('edge [fontname="%s"] ;' % self.fontname)

        if self.rotate:
            self.out_file.write("rankdir=LR ;")

    def recurse(self, tree, node_id, criterion, decision_tree, parent=None, depth=0):
        original_data = decision_tree.original_data
        test_data = decision_tree.test_data
        if node_id == -1:  # _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % -1)  # _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if self.max_depth is None or depth <= self.max_depth:

            # Collect ranks for 'leaf' option in plot_options
            if left_child == -1:  # _tree.TREE_LEAF:
                self.ranks["leaves"].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))

            headers = "{   | # data | deceased | P_hat(C1) }"  # | missing data | missing feature | P(misclass)|" \
            #" AUC | Sensitivity | Specificity | threshold

            # Get values to insert in graph
            values = self.node_to_str(tree, node_id, criterion=criterion, decision_tree=decision_tree, parent=parent)
            table_string1 = f"|{{data_test | " \
                            f"{values.loc['# patients', 'tree_train'] } | " \
                            f"{values.loc['deceased', 'tree_train']} |" \
                            f"{values.loc['prediction', 'tree_train']} " \
                            f"}}"
                            #f"{values.loc['Missing data', 'tree_train']} |" \
                            #f"{values.loc['Missing feature', 'tree_train']} |" \
                            #f"{values.loc['misclassified', 'tree_train']} |" \
                            #f"{values.loc['auc', 'tree_train']} |" \
                            #f"{values.loc['sensitivity', 'tree_train']} |" \
                            #f"{values.loc['specificity', 'tree_train']} |" \
                            #f"{values.loc['threshold', 'tree_train']} " \

            table_string2 = f"|{{data_train_sl | " \
                            f"{values.loc['# patients', 'sl_train']} | " \
                            f"{values.loc['deceased', 'sl_train']} |" \
                            f"{values.loc['prediction', 'sl_train']} " \
                            f"}}" if not (values['tree_train'] == values['sl_train']).all() else ''
                            #f"{values.loc['Missing data', 'sl_train']} |" \
                            #f"{values.loc['Missing feature', 'sl_train']} |" \
                            #f"{values.loc['misclassified', 'sl_train']} |" \
                            #f"{values.loc['auc', 'sl_train']} |" \
                            #f"{values.loc['sensitivity', 'sl_train']} |" \
                            #f"{values.loc['specificity', 'sl_train']} |" \
                            #f"{values.loc['threshold', 'sl_train']} " \

            feature_row = f'| {self.feature_names[tree.feature[node_id]]} \<= ' \
                          f'{round(tree.threshold[node_id], 2)}' if left_child != -1 else ''

            self.out_file.write(
                '%d [label="{{%s %s}%s}"' % (node_id, headers, table_string1 + table_string2, feature_row)
            )
            #self.out_file.write(
            #   ' \n%d [label = "{{{      | # data | P(C1) | P_hat(C1) | missing data | missing feature | P(misclass) } | {data_test | nb1 | p1 | ph1 | m1 | mf1 | Pm1} | {data_train | nb2 | p2 | ph2 | m2 | mf2 | Pm2}} | Feature \<= thresh}  "];' % (node_id)  #'%d [label="{{%s %s}| %s}"' % (node_id, headers, test, 'feature = xy')
            #)


            #self.out_file.write(
            #    '%d [label = "{{lel} %s}"' % (node_id, feature_row)
            #)

            if self.filled:
                self.out_file.write(
                    ', fillcolor="%s"' % self.get_fill_color(tree, node_id)
                )
            self.out_file.write("];")

            if parent is not None:
                # Add edge to parent
                self.out_file.write("%d -> %d" % (parent, node_id))
                if parent >= 0:  # == 0:  Changed so it always draws True/False labels
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    # if left_child > 0:
                    # self.out_file.write(" [labeldistance=2.5, labelangle=")
                    # self.out_file.write(f'%d, headlabel="'
                    # f'{original_data[left_child].shape[0]}"]' % angles[0])

                    self.out_file.write(" [labeldistance= 2, labelangle=")

                    """feature = self.feature_names[tree.feature[parent]]
                    missing_perc1 = round(original_data[node_id][f'miss_{feature}'].mean() * 100, 2)
                    missing_perc2 = round(test_data[node_id][f'miss_{feature}'].mean() * 100, 2)
                    """
                    if node_id in tree.children_left:  # == 1:

                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])  # -{missing_perc1}%%-{missing_perc2}%%
                self.out_file.write(" ;\n")

            if left_child != -1:  # _tree.TREE_LEAF:
                self.recurse(
                    tree,
                    left_child,
                    criterion=criterion,
                    decision_tree=decision_tree,
                    parent=node_id,
                    depth=depth + 1,
                )
                self.recurse(
                    tree,
                    right_child,
                    criterion=criterion,
                    decision_tree=decision_tree,
                    parent=node_id,
                    depth=depth + 1,
                )

        else:
            self.ranks["leaves"].append(str(node_id))

            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                # color cropped nodes grey
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write("] ;\n" % node_id)

            if parent is not None:
                # Add edge to parent
                self.out_file.write("%d -> %d ;\n" % (parent, node_id))


def export_graphviz_tree(
        decision_tree,
        original_data,
        test_data,
        out_file=None,
        *,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        leaves_parallel=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rotate=False,
        rounded=False,
        special_characters=False,
        precision=3,
        fontname="helvetica",
):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : object or str, default=None
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

        .. versionchanged:: 0.20
            Default of out_file changed from "tree.dot" to None.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of str, default=None
        Names of each of the features.
        If None generic names will be used ("feature_0", "feature_1", ...).

    class_names : list of str or bool, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, default=False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, default=False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners.

    special_characters : bool, default=False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontname : str, default='helvetica'
        Name of font used to render text.

    Returns
    -------
    dot_data : str
        String representation of the input tree in GraphViz dot format.
        Only returned if ``out_file`` is None.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf)
    'digraph Tree {...
    """

    own_file = False
    return_string = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True

        if out_file is None:
            return_string = True
            out_file = StringIO()

        exporter = _DOTTreeExporter(
            out_file=out_file,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            leaves_parallel=leaves_parallel,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rotate=rotate,
            rounded=rounded,
            special_characters=special_characters,
            precision=precision,
            fontname=fontname,
        )
        decision_tree.original_data = [None] * decision_tree.tree_.node_count
        decision_tree.original_data[0] = original_data

        decision_tree.features_splits = [''] * decision_tree.tree_.node_count

        decision_tree.test_data = [None] * decision_tree.tree_.node_count
        decision_tree.test_data[0] = test_data

        exporter.export(decision_tree)

        if return_string:
            return exporter.out_file.getvalue()

    finally:
        if own_file:
            out_file.close()
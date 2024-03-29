import inspect
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score


def filter_dict(func, **kwarg_dict):
    """
    Function to filter **kwargs arguments input to only keep relevant arguments for a function
    :param func:
    :param kwarg_dict:
    :return:
    """
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    # if 'kwargs' in sign:
    #    return kwarg_dict

    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}

    return filtered_dict


def get_mdr(Y_target, Y_predicted, predicted_accuracies):
    unique_accuracies = np.sort(np.unique(predicted_accuracies))[::-1]

    mdr_values = []

    for perc_acc in unique_accuracies:
        dr = sum(predicted_accuracies >= perc_acc) / len(Y_target)
        auc = roc_auc_score(Y_target[predicted_accuracies >= perc_acc],
                            Y_predicted[predicted_accuracies >= perc_acc]) if \
            len(np.unique(Y_target[predicted_accuracies >= perc_acc])) > 1 else 0
        perc_node = sum(predicted_accuracies >= perc_acc) / len(Y_target)
        acc = accuracy_score(Y_target[predicted_accuracies >= perc_acc],
                             Y_predicted[predicted_accuracies >= perc_acc])
        bal_acc = balanced_accuracy_score(Y_target[predicted_accuracies >= perc_acc],
                                          Y_predicted[predicted_accuracies >= perc_acc])
        sensitivity = recall_score(Y_target[predicted_accuracies >= perc_acc],
                                   Y_predicted[predicted_accuracies >= perc_acc]
                                   , pos_label=1, zero_division=0)
        specificity = recall_score(Y_target[predicted_accuracies >= perc_acc],
                                   Y_predicted[predicted_accuracies >= perc_acc]
                                   , pos_label=0, zero_division=0)
        mdr_values.append({'dr': dr, 'accuracy': acc, 'bal_acc': bal_acc,
                           'sens': sensitivity, 'spec': specificity, 'auc': auc})

    mdr_values = np.array(mdr_values)

    return mdr_values

from sapsii.saps_processing import convert_saps
import pandas as pd
import numpy as np

import os
from copy import deepcopy
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, recall_score

from mdr_figure_maker import generate_mdr
from models import XGBClassifier, SapsModel

# Constants
pos_label = 1
y_true_str = 'y_true'
all_metrics = None
UNFIXED = "unfixed/"
FIXED_APC = "fixedapc/"
FIXED_ALL = "fixedall/"


def save_mdr_metrics(mdr_curves, df_name, fixed_ipc, fixed_apc):
    global all_metrics

    metrics_vals = {'df': df_name, 'fixed_ipc': fixed_ipc, 'fixed_apc': fixed_apc}
    metrics_names = ["Acc", "Bal_Acc", "sens", "spec", "Auc", "Auprc", "PPV", "NPV"]
    for metric_name in metrics_names:
        metrics_vals[metric_name] = round(mdr_curves[metric_name][0], 3)

    auc_mins = [perc / 100 for perc in range(80, 96)]
    for auc_min in auc_mins:
        auc_index = -1
        for index, element in enumerate(mdr_curves["Auc"]):
            if element >= auc_min:
                auc_index = index
                break
        min_dr = 0 if auc_index == -1 else mdr_curves["DR"][auc_index]
        metrics_vals[f'DR-auc{auc_min}'] = min_dr
    all_metrics = pd.concat([all_metrics, pd.DataFrame([metrics_vals])], ignore_index=True)


def produce_results(df_mimic, df_eicu, model, path_save, threshold_correction, calibrate,
                    saps_transform=False, class_weighting=True, do_eicu=True):
    global pos_label, y_true_str, UNFIXED, FIXED_ALL, FIXED_APC, all_metrics

    all_metrics = pd.DataFrame()

    df_mimic = deepcopy(df_mimic)
    df_eicu = deepcopy(df_eicu)

    # Save parameters
    experiment_params = {'index': [0],
                         'model': model,
                         'threshold_correction': threshold_correction,
                         'saps_transform': saps_transform,
                         'class_weighting': class_weighting}

    # get mimic df
    df_0 = df_mimic[(df_mimic['anchor_year_group'] == '2008 - 2010') | (df_mimic['anchor_year_group'] == '2011 - 2013')]
    df_0 = df_0.drop(columns=['anchor_year_group'])

    df_0_valid = df_0.sample(frac=0.4, random_state=200)
    df_0 = df_0.drop(df_0_valid.index)

    df_0_valid = df_0_valid.reset_index(drop=True)
    df_0 = df_0.reset_index(drop=True)

    df_1 = df_mimic[df_mimic['anchor_year_group'] == '2014 - 2016']
    df_1 = df_1.drop(columns=['anchor_year_group'])
    df_2 = df_mimic[df_mimic['anchor_year_group'] == '2017 - 2019']
    df_2 = df_2.drop(columns=['anchor_year_group'])

    # Get outcome
    y_0 = np.array(df_0.pop('deceased'))
    y_0_valid = np.array(df_0_valid.pop('deceased'))
    y_1 = np.array(df_1.pop('deceased'))
    y_2 = np.array(df_2.pop('deceased'))
    y_eicu = np.array(df_eicu.pop('deceased'))

    # Get knnimputed values
    imputer = KNNImputer(n_neighbors=20)
    df_0 = pd.DataFrame(imputer.fit_transform(df_0), columns=df_0.columns)

    df_0_valid = pd.DataFrame(imputer.transform(df_0_valid), columns=df_0_valid.columns)
    df_1 = pd.DataFrame(imputer.transform(df_1), columns=df_1.columns)
    df_2 = pd.DataFrame(imputer.transform(df_2), columns=df_2.columns)
    df_eicu.loc[:, df_0.columns] = imputer.transform(df_eicu.loc[:, df_0.columns])

    # Round variables
    variables_to_round = ['mets', 'hem', 'aids', 'cpap', 'vent']
    df_0[variables_to_round] = df_0[variables_to_round].round()
    df_0_valid[variables_to_round] = df_0_valid[variables_to_round].round()
    df_eicu[variables_to_round] = df_eicu[variables_to_round].round()
    df_1[variables_to_round] = df_1[variables_to_round].round()
    df_2[variables_to_round] = df_2[variables_to_round].round()

    # Apply saps
    if saps_transform or model == 'saps':
        df_0_score, _ = convert_saps(df_0)
        df_0_valid_score, _ = convert_saps(df_0_valid)
        df_1_score, _ = convert_saps(df_1)
        df_2_score, _ = convert_saps(df_2)
        df_eicu_score, _ = convert_saps(df_eicu)
    else:
        _, df_0_score = convert_saps(df_0)
        _, df_0_valid_score = convert_saps(df_0_valid)
        _, df_1_score = convert_saps(df_1)
        _, df_2_score = convert_saps(df_2)
        _, df_eicu_score = convert_saps(df_eicu)

    # Train model
    # Parameter grid
    param_grid = {
        'max_depth': range(2, min(int(np.log2(df_0_score.shape[0])) + 1, 11))  #,
        #'min_samples_leaf': [5, 10, 15]
    }
    # Base model
    params = {}
    if model == 'xgb':
        clf = XGBClassifier(objective='binary:logistic', random_state=54288, class_weighting=class_weighting)
        params = {'n_trials': 150}
    elif model == 'saps':
        clf = SapsModel()

    clf.fit(df_0_score, y_0, threshold=threshold_correction, calibrate=calibrate, **params)

    experiment_params['threshold'] = clf.THRESHOLD

    path = os.path.abspath(path_save + "calibration.pdf")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    clf.show_calibration(data=df_0_valid_score, target=y_0_valid, save_path=path_save + "calibration.pdf")
    df = pd.DataFrame.from_dict(experiment_params)
    df.to_csv(path_save + "params.csv", index=False)

    # Validation metrics
    y_0_valid_prob = clf.predict_proba(df_0_valid_score)[:, 1]

    # MIMIC 2014 metrics
    y_1_prob = clf.predict_proba(df_1_score)[:, 1]

    # MIMIC 2017 metrics
    y_2_prob = clf.predict_proba(df_2_score)[:, 1]

    # eICU metrics
    df_eicu_score['prediction'] = clf.predict(df_eicu_score.loc[:, df_0_score.columns])
    df_eicu_score['probability'] = clf.predict_proba(df_eicu_score.loc[:, df_0_score.columns])[:, 1]
    df_eicu_score['y_true'] = y_eicu

    hosp_id = np.unique(df_eicu['hospitalid'])

    # Generate MDR
    hosp_class_imbalance = len(y_0_valid[y_0_valid == pos_label]) / len(y_0_valid)
    fitted_models, mdr_valid = generate_mdr(x=df_0_valid_score, y=y_0_valid, predicted_prob=y_0_valid_prob,
                                            pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                            filename=path_save + UNFIXED + "mimic_valid_2008",
                                            split_valid=True, return_infos=True, threshold=clf.THRESHOLD)
    fixed_tree = fitted_models['apc']
    fixed_ipc = fitted_models['ipc']
    save_mdr_metrics(mdr_valid, "MIMICValid", fixed_ipc=False, fixed_apc=False)

    if do_eicu:
        _, mdr_valid = generate_mdr(x=df_0_valid_score, y=y_0_valid, predicted_prob=y_0_valid_prob,
                                    pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                    filename=path_save + FIXED_ALL + "mimic_valid_2008",
                                    split_valid=True, return_infos=True,
                                    fixed_tree=fixed_tree, fixed_ipc=fixed_ipc, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_valid, "MIMICValid", fixed_ipc=True, fixed_apc=True)

        # MIMIC 2008 (50%) & MIMIC 2014 (50%)
        df_0_half = df_0_valid_score.sample(frac=0.5, random_state=54288)
        df_1_half = df_1_score.sample(frac=0.5, random_state=54288)

        y_01 = np.append(y_0_valid[df_0_half.index], y_1[df_1_half.index])
        y_01_prob = np.append(y_0_valid_prob[df_0_half.index], y_1_prob[df_1_half.index])
        df_01_score = pd.concat([df_0_half, df_1_half], ignore_index=True)

        hosp_class_imbalance = len(y_01[y_01 == pos_label]) / len(y_01)
        _, mdr_01 = generate_mdr(x=df_01_score, y=y_01, predicted_prob=y_01_prob,
                                 pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                 filename=path_save + UNFIXED + "mimic_0814",
                                 split_valid=True, fixed_tree=None, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_01, 'MIMIC0814', fixed_ipc=False, fixed_apc=False)

        _, mdr_01 = generate_mdr(x=df_01_score, y=y_01, predicted_prob=y_01_prob,
                                 pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                 filename=path_save + FIXED_APC + "mimic_0814",
                                 split_valid=True, fixed_tree=fixed_tree, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_01, 'MIMIC0814', fixed_ipc=False, fixed_apc=True)

        _, mdr_01 = generate_mdr(x=df_01_score, y=y_01, predicted_prob=y_01_prob,
                                 pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                 filename=path_save + FIXED_ALL + "mimic_0814",
                                 split_valid=False, fixed_tree=fixed_tree, fixed_ipc=fixed_ipc, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_01, 'MIMIC0814', fixed_ipc=True, fixed_apc=True)

    # MIMIC 2014
    hosp_class_imbalance = len(y_1[y_1 == pos_label]) / len(y_1)
    _, mdr_1 = generate_mdr(x=df_1_score, y=y_1, predicted_prob=y_1_prob,
                            pos_class_weight=1 - round(hosp_class_imbalance, 2),
                            filename=path_save + UNFIXED + "mimic_2014",
                            split_valid=True, fixed_tree=None, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
    save_mdr_metrics(mdr_1, 'MIMIC2014', fixed_ipc=False, fixed_apc=False)

    if do_eicu:
        _, mdr_1 = generate_mdr(x=df_1_score, y=y_1, predicted_prob=y_1_prob,
                                pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                filename=path_save + FIXED_APC + "mimic_2014",
                                split_valid=True, fixed_tree=fixed_tree, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_1, 'MIMIC2014', fixed_ipc=False, fixed_apc=True)

        _, mdr_1 = generate_mdr(x=df_1_score, y=y_1, predicted_prob=y_1_prob,
                                pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                filename=path_save + FIXED_ALL + "mimic_2014",
                                split_valid=False, fixed_tree=fixed_tree, fixed_ipc=fixed_ipc, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_1, 'MIMIC2014', fixed_ipc=True, fixed_apc=True)

        # MIMIC 2014 (50%) & MIMIC 2017 (50%)
        df_1_otherhalf = df_1_score.loc[~df_1_score.index.isin(df_1_half.index)]
        df_2_half = df_2_score.sample(frac=0.5, random_state=54288)

        y_12 = np.append(y_1[df_1_otherhalf.index], y_2[df_2_half.index])
        y_12_prob = np.append(y_1_prob[df_1_otherhalf.index], y_2_prob[df_2_half.index])
        df_12_score = pd.concat([df_1_otherhalf, df_2_half], ignore_index=True)

        hosp_class_imbalance = len(y_01[y_01 == pos_label]) / len(y_01)

        _, mdr_01 = generate_mdr(x=df_12_score, y=y_12, predicted_prob=y_12_prob,
                                 pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                 filename=path_save + UNFIXED + "mimic_1417",
                                 split_valid=True, fixed_tree=None, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_01, 'MIMIC1417', fixed_ipc=False, fixed_apc=False)

        _, mdr_01 = generate_mdr(x=df_12_score, y=y_12, predicted_prob=y_12_prob,
                                 pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                 filename=path_save + FIXED_APC + "mimic_1417",
                                 split_valid=True, fixed_tree=fixed_tree, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_01, 'MIMIC1417', fixed_ipc=False, fixed_apc=True)

        _, mdr_01 = generate_mdr(x=df_12_score, y=y_12, predicted_prob=y_12_prob,
                                 pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                 filename=path_save + FIXED_ALL + "mimic_1417",
                                 split_valid=False, fixed_tree=fixed_tree, fixed_ipc=fixed_ipc, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_01, 'MIMIC1417', fixed_ipc=True, fixed_apc=True)

    # MIMIC 2017
    hosp_class_imbalance = len(y_2[y_2 == pos_label]) / len(y_2)

    _, mdr_2 = generate_mdr(x=df_2_score, y=y_2, predicted_prob=y_2_prob,
                            pos_class_weight=1 - round(hosp_class_imbalance, 2),
                            filename=path_save + UNFIXED + "mimic_2017",
                            split_valid=True, fixed_tree=None, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
    save_mdr_metrics(mdr_2, "MIMIC2017", fixed_ipc=False, fixed_apc=False)

    if do_eicu:
        _, mdr_2 = generate_mdr(x=df_2_score, y=y_2, predicted_prob=y_2_prob,
                                pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                filename=path_save + FIXED_APC + "mimic_2017",
                                split_valid=True, fixed_tree=fixed_tree, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_2, "MIMIC2017", fixed_ipc=False, fixed_apc=True)

        _, mdr_2 = generate_mdr(x=df_2_score, y=y_2, predicted_prob=y_2_prob,
                                pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                filename=path_save + FIXED_ALL + "mimic_2017",
                                split_valid=False, fixed_tree=fixed_tree, fixed_ipc=fixed_ipc, return_infos=True, threshold=clf.THRESHOLD)
        save_mdr_metrics(mdr_2, "MIMIC2017", fixed_ipc=True, fixed_apc=True)

    # eICU
    if do_eicu:
        for hid in hosp_id:
            df_hid = df_eicu_score[df_eicu_score['hospitalid'] == hid].drop(columns=['hospitalid']).reset_index(drop=True)
            if df_hid.shape[0] >= 200:
                # df_hid.to_csv(f'../../../data/mimic_mimic/meicu/df_m0_eicu{hid}.csv', index=False)

                hosp_y = df_hid.pop(y_true_str).to_numpy()  # df_hid[y_true_str].to_numpy()
                hosp_y_prob = df_hid.pop('probability').to_numpy()
                hosp_y_pred = df_hid.pop('prediction').to_numpy()

                # Save metrics for specific hospital
                # save_metrics(y_true=hosp_y, y_pred=hosp_y_pred, y_prob=hosp_y_prob, df_name='eicu_' + str(hid))

                hosp_filename = path_save + 'meicu/hosp_' + str(hid) + '/'
                hosp_class_imbalance = len(hosp_y[hosp_y == pos_label]) / len(hosp_y)

                _, mdr_hosp = generate_mdr(x=df_hid, y=hosp_y, predicted_prob=hosp_y_prob,
                                           pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                           filename=hosp_filename + UNFIXED[:-1],
                                           split_valid=True,
                                           fixed_tree=None, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
                save_mdr_metrics(mdr_hosp, f"eICU_{hid}", fixed_ipc=False, fixed_apc=False)

                _, mdr_hosp = generate_mdr(x=df_hid, y=hosp_y, predicted_prob=hosp_y_prob,
                                           pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                           filename=hosp_filename + FIXED_APC[:-1],
                                           split_valid=True,
                                           fixed_tree=fixed_tree, fixed_ipc=None, return_infos=True, threshold=clf.THRESHOLD)
                save_mdr_metrics(mdr_hosp, f"eICU_{hid}", fixed_ipc=False, fixed_apc=True)

                _, mdr_hosp = generate_mdr(x=df_hid, y=hosp_y, predicted_prob=hosp_y_prob,
                                           pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                           filename=hosp_filename + FIXED_ALL[:-1],
                                           split_valid=False,
                                           fixed_tree=fixed_tree, fixed_ipc=fixed_ipc, return_infos=True, threshold=clf.THRESHOLD)
                save_mdr_metrics(mdr_hosp, f"eICU_{hid}", fixed_ipc=True, fixed_apc=True)

    all_metrics.to_csv(path_save + "results.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("../../data/sapsii/mimic_filtered_data.csv")
    df = df.drop(columns=['stay_id', 'hospitalid'])

    # df_eicu = pd.read_csv("../../data/sapsii/eicu_filtered_data.csv")
    # df_eicu = df_eicu.drop(columns=['stay_id'])
    # produce_results(df_mimic=df, df_eicu=df_eicu)

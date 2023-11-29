from sapsii.saps_processing import convert_saps
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, recall_score

from mdr_figure_maker import generate_mdr

# Constants
FIXED_TREE = True
pos_label = 1
y_true_str = 'y_true'
saved_files = 'hosp/1129/'
all_metrics = pd.DataFrame()


def save_metrics(y_true, y_pred, y_prob, df_name):
    global all_metrics
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics["Bal Acc"] = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics["AUC"] = roc_auc_score(y_true=y_true, y_score=y_prob)
    metrics["Sensitivity"] = recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)
    metrics["Specificity"] = recall_score(y_true=y_true, y_pred=y_pred, pos_label=0)
    all_metrics[df_name] = metrics


def save_mdr_metrics(mdr_curves, df_name):
    global all_metrics

    metrics_vals = {'df': df_name}
    metrics_names = ["Acc", "Bal_Acc", "sens", "spec", "Auc", "Auprc"]
    for metric_name in metrics_names:
        metrics_vals[metric_name] = mdr_curves[metric_name][0]

    auc_mins = [perc/100 for perc in range(80, 96)]
    for auc_min in auc_mins:
        auc_index = -1
        for index, element in enumerate(mdr_curves["Auc"]):
            if element >= auc_min:
                auc_index = index
                break
        min_dr = 0 if auc_index == -1 else mdr_curves["DR"][auc_index]
        metrics_vals[f'DR-auc{auc_min}'] = min_dr
    all_metrics = pd.concat([all_metrics, pd.DataFrame([metrics_vals])], ignore_index=True)


def produce_results(df_mimic, df_eicu):
    global FIXED_TREE, pos_label, y_true_str, saved_files

    # get mimic df
    df_0 = df_mimic[(df_mimic['anchor_year_group'] == '2008 - 2010') | (df_mimic['anchor_year_group'] == '2011 - 2013')]
    df_0 = df_0.drop(columns=['anchor_year_group'])

    df_0_valid = df_0.sample(frac=0.2, random_state=200)
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
    df_0_score, _ = convert_saps(df_0)
    df_0_valid_score, _ = convert_saps(df_0_valid)
    df_1_score, _ = convert_saps(df_1)
    df_2_score, _ = convert_saps(df_2)
    df_eicu_score, _ = convert_saps(df_eicu)

    # Train model
    # Parameter grid
    param_grid = {
        'max_depth': range(2, min(int(np.log2(df_0_score.shape[0])) + 1, 11)),
        'min_samples_leaf': [5, 10, 15]
    }
    # Base model
    clf = RandomForestClassifier(random_state=54288, class_weight="balanced")

    # Instantiate the grid search model
    print('Hyperparameter optimization')
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               cv=min(4, int(df_0_score.shape[0] / 2)), n_jobs=-1, verbose=0)

    # Fit the grid search to the data
    grid_search.fit(df_0_score, y_0)
    print(grid_search.best_params_)

    # Get best model
    clf = grid_search.best_estimator_

    # clf = RandomForestClassifier().fit(df_0_score, y_0)

    # Get training metrics
    y_0_pred = clf.predict(df_0_score)
    y_0_prob = clf.predict_proba(df_0_score)[:, 1]
    #save_metrics(y_true=y_0, y_pred=y_0_pred, y_prob=y_0_prob, df_name="Train")

    # Validation metrics
    y_0_valid_pred = clf.predict(df_0_valid_score)
    y_0_valid_prob = clf.predict_proba(df_0_valid_score)[:, 1]
    #save_metrics(y_true=y_0_valid, y_pred=y_0_valid_pred, y_prob=y_0_valid_prob, df_name="Validation")

    # MIMIC 2014 metrics
    y_1_pred = clf.predict(df_1_score)
    y_1_prob = clf.predict_proba(df_1_score)[:, 1]
    #save_metrics(y_true=y_1, y_pred=y_1_pred, y_prob=y_1_prob, df_name="MIMIC2014")

    # MIMIC 2017 metrics
    y_2_pred = clf.predict(df_2_score)
    y_2_prob = clf.predict_proba(df_2_score)[:, 1]
    #save_metrics(y_true=y_2, y_pred=y_2_pred, y_prob=y_2_prob, df_name="MIMIC2017")

    # eICU metrics
    df_eicu_score['prediction'] = clf.predict(df_eicu_score.loc[:, df_0_score.columns])
    df_eicu_score['probability'] = clf.predict_proba(df_eicu_score.loc[:, df_0_score.columns])[:, 1]
    df_eicu_score['y_true'] = y_eicu
    #save_metrics(y_true=y_eicu, y_pred=df_eicu_score['prediction'], y_prob=df_eicu_score['probability'], df_name="eICU")

    hosp_id = np.unique(df_eicu['hospitalid'])

    # Generate MDR
    hosp_class_imbalance = len(y_0_valid[y_0_valid == pos_label]) / len(y_0_valid)
    fixed_tree, mdr_valid = generate_mdr(x=df_0_valid_score, y=y_0_valid, predicted_prob=y_0_valid_prob,
                                         pos_class_weight=1 - round(hosp_class_imbalance, 2),
                                         filename=saved_files + "mimic_valid_2008",
                                         split_valid=False, return_infos=True)
    save_mdr_metrics(mdr_valid, "MIMICValid")

    if not FIXED_TREE:
        fixed_tree = None

    # MIMIC 2014
    hosp_class_imbalance = len(y_1[y_1 == pos_label]) / len(y_1)
    _, mdr_1 = generate_mdr(x=df_1_score, y=y_1, predicted_prob=y_1_prob,
                            pos_class_weight=1 - round(hosp_class_imbalance, 2), filename=saved_files + "mimic_2014",
                            fixed_tree=fixed_tree, return_infos=True)

    save_mdr_metrics(mdr_1, 'MIMIC2014')

    # MIMIC 2017
    hosp_class_imbalance = len(y_2[y_2 == pos_label]) / len(y_2)
    _, mdr_2 = generate_mdr(x=df_2_score, y=y_2, predicted_prob=y_2_prob,
                            pos_class_weight=1 - round(hosp_class_imbalance, 2), filename=saved_files + "mimic_2017",
                            fixed_tree=fixed_tree, return_infos=True)
    save_mdr_metrics(mdr_2, "MIMIC2017")

    # eICU
    for hid in hosp_id:
        df_hid = df_eicu_score[df_eicu_score['hospitalid'] == hid].drop(columns=['hospitalid']).reset_index(drop=True)
        if df_hid.shape[0] >= 200:
            # df_hid.to_csv(f'../../../data/mimic_mimic/meicu/df_m0_eicu{hid}.csv', index=False)

            hosp_y = df_hid.pop(y_true_str).to_numpy()  # df_hid[y_true_str].to_numpy()
            hosp_y_prob = df_hid.pop('probability').to_numpy()
            hosp_y_pred = df_hid.pop('prediction').to_numpy()

            # Save metrics for specific hospital
            #save_metrics(y_true=hosp_y, y_pred=hosp_y_pred, y_prob=hosp_y_prob, df_name='eicu_' + str(hid))

            hosp_filename = saved_files + 'meicu/hosp_' + str(hid)
            hosp_class_imbalance = len(hosp_y[hosp_y == pos_label]) / len(hosp_y)

            _, mdr_hosp = generate_mdr(x=df_hid, y=hosp_y, predicted_prob=hosp_y_prob,
                                       pos_class_weight=1 - round(hosp_class_imbalance, 2), filename=hosp_filename,
                                       split_valid=False, fixed_tree=fixed_tree, return_infos=True)
            save_mdr_metrics(mdr_hosp, f"eICU_{hid}")
    all_metrics.to_csv(saved_files + "results.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("../../data/sapsii/mimic_filtered_data.csv")
    df = df.drop(columns=['stay_id', 'hospitalid'])

    df_eicu = pd.read_csv("../../data/sapsii/eicu_filtered_data.csv")
    df_eicu = df_eicu.drop(columns=['stay_id'])
    produce_results(df_mimic=df, df_eicu=df_eicu)

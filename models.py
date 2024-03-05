import ml_insights as mli
import numpy as np
import optuna
import xgboost as xgb

import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import balanced_accuracy_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import class_weight


class BaseModel:
    clf = None
    THRESHOLD = 0.5
    calibration = None
    random_state = None
    classes_ = [0, 1]  # To allow model calibration

    def __init__(self):
        raise NotImplementedError

    def calibrate_model(self, y_pred, y_true, data=None, method='sklearn'):
        if method == 'sklearn':
            calibration = CalibratedClassifierCV(estimator=deepcopy(self), method='isotonic', cv='prefit')
            calibration.fit(data, y_true)
        elif method == 'spline':
            calibration = mli.SplineCalib(random_state=self.random_state, method='liblinear')
            calibration.fit(y_model=y_pred, y_true=y_true)
        else:
            raise NotImplementedError
        self.calibration = calibration

    def fit(self):
        raise NotImplementedError

    def predict_proba(self, data):
        raise NotImplementedError

    def predict(self, data):
        return (self.predict_proba(data)[:, 1] >= self.THRESHOLD).astype(int)

    def show_calibration(self, data, target, save_path=None):
        predicted_prob = self.predict_proba(data)[:, 1]
        CalibrationDisplay.from_predictions(target, predicted_prob)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def _optimal_threshold_auc(target, predicted):
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]

    @staticmethod
    def _optimal_threshold_auprc(target, predicted):
        precision, recall, threshold = precision_recall_curve(target, predicted)
        # Remove last element
        precision = precision[:-1]
        recall = recall[:-1]

        i = np.arange(len(recall))
        roc = pd.DataFrame({'tf': pd.Series(precision * recall, index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]


class XGBClassifier(BaseModel):

    def __init__(self, objective='binary:logistic', random_state=None, class_weighting=False):
        self.objective = objective
        self.random_state = random_state
        self.class_weighting = class_weighting

    def fit(self, data, target, n_trials=100, timeout: int = None, threshold: str = None, calibrate: bool = False):
        if self.random_state:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
        else:
            study = optuna.create_study(direction="maximize")

        if True:
            data = deepcopy(data)
            target = deepcopy(target)
            data['target'] = target
            calibration_data = data.sample(frac=0.3, random_state=self.random_state)
            data = data.drop(calibration_data.index)
            calibration_target = calibration_data['target']
            calibration_data = calibration_data.drop(columns=['target'])
            target = data['target']
            data = data.drop(columns=['target'])

        study.optimize(self._objective(data, target), n_trials=n_trials, timeout=timeout)
        best_trial = study.best_trial
        params = best_trial.params
        params['objective'] = self.objective
        self.clf = xgb.XGBClassifier(**params)

        if self.class_weighting:
            samples_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=target
            )
            self.clf.fit(data, target, sample_weight=samples_weights)
        else:
            self.clf.fit(data, target)

        if calibrate:
            self.calibrate_model(y_pred=self.clf.predict_proba(calibration_data)[:, 1],
                                 y_true=calibration_target,
                                 data=calibration_data)

        if threshold:
            predicted = self.clf.predict_proba(data)[:, 1]
            if threshold.lower() == 'auc':
                self.THRESHOLD = self._optimal_threshold_auc(target=target, predicted=predicted)
            elif threshold.lower() == 'auprc':
                self.THRESHOLD = self._optimal_threshold_auprc(target=target, predicted=predicted)
            else:
                raise NotImplementedError

    def predict_proba(self, X):
        if self.calibration:
            #calibrated_predictions = self.calibration.predict_proba(probability[:, 1])
            probability = self.calibration.predict_proba(X)#np.array([1 - calibrated_predictions, calibrated_predictions]).transpose()
        else:
            probability = self.clf.predict_proba(X)
        return probability

    def _objective(self, data, target):
            def __objective(trial):
                # train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
                # dtrain = xgb.DMatrix(train_x, label=train_y)
                # dvalid = xgb.DMatrix(valid_x, label=valid_y)

                param = {
                    "device": "gpu",
                    "verbosity": 0,
                    "objective": self.objective,
                    # use exact for small dataset.
                    "tree_method": "exact",
                    # defines booster, gblinear for linear functions.
                    "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                    # L2 regularization weight.
                    "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                    # L1 regularization weight.
                    "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                    # sampling ratio for training data.
                    "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                    # sampling according to each tree.
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                }

                if param["booster"] in ["gbtree", "dart"]:
                    # maximum depth of the tree, signifies complexity of the tree.
                    param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                    # minimum child weight, larger the term more conservative the tree.
                    param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                    param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                    # defines how selective algorithm is.
                    param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                    param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

                if param["booster"] == "dart":
                    param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                    param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                    param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                    param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

                if self.class_weighting:
                    samples_weights = class_weight.compute_sample_weight(
                        class_weight='balanced',
                        y=target
                    )
                else:
                    samples_weights = None

                metric = np.mean(cross_val_score(xgb.XGBClassifier(**param), data, target,
                                                 scoring='roc_auc',
                                                 fit_params={'sample_weight': samples_weights}))

                # param['sample_weight'] = samples_weights

                # bst = xgb.train(param, dtrain)  # )
                # bst = xgb.XGBClassifier(**param).fit(train_x, train_y, sample_weight=samples_weights)
                # preds = bst.predict(valid_x)
                # pred_labels = np.rint(preds)
                # metric = balanced_accuracy_score(valid_y, pred_labels)  # roc_auc_score(valid_y, preds)
                return metric

            return __objective


class SapsModel(BaseModel):

    def __init__(self):
        pass

    def fit(self, data, target, threshold=None, calibrate: bool = False):
        if calibrate:
            self.calibrate_model(y_pred=self.predict_proba(data)[:, 1], y_true=target,
                                 data=data)

        if threshold:
            predicted = self.predict_proba(data)[:, 1]
            if threshold.lower() == 'auc':
                self.THRESHOLD = self._optimal_threshold_auc(target=target, predicted=predicted)
            elif threshold.lower() == 'auprc':
                self.THRESHOLD = self._optimal_threshold_auprc(target=target, predicted=predicted)
            else:
                raise NotImplementedError
        print(f'{self.THRESHOLD=}')

    def predict_proba(self, X):
        if self.calibration:
            probability = self.calibration.predict_proba(X)
        else:
            score = X.sum(axis=1)
            logit = -7.7631 + 0.0737 * score + 0.9971 * np.log(score + 1)
            probability = np.exp(logit) / (1 + np.exp(logit))
            probability = np.array([1 - probability, probability]).transpose()

        return probability

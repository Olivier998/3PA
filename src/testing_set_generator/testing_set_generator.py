"""

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score, matthews_corrcoef, \
    precision_score, f1_score
from sklearn.model_selection import GridSearchCV

from settings.paths import Paths


def testing_set_generator():
    # Import and define datasets
    mimic_df = pd.read_csv(Paths.MIMIC_SAPS_VARIABLES)
    mimic_train = Dataset(mimic_df[mimic_df['anchor_year_group'].isin(['2008 - 2010', '2011 - 2013'])])
    mimic_test1 = Dataset(mimic_df[mimic_df['anchor_year_group'] == '2014 - 2016'], title="Mimic_14-16")
    mimic_test2 = Dataset(mimic_df[mimic_df['anchor_year_group'] == '2017 - 2019'], title="Mimic_17-19")
    del mimic_df

    eicu_df = pd.read_csv(Paths.EICU_SAPS_VARIABLES)
    eicu_test = Dataset(eicu_df, title="eICU")
    del eicu_df

    # Data imputation
    print("Data imputation")
    imputer = Imputer(mimic_train)
    imputer.impute(mimic_test1)
    imputer.impute(mimic_test2)
    imputer.impute(eicu_test)

    # Train model
    # Parameter grid
    param_grid = {
        'max_depth': range(2, 8)  # int(np.log2(mimic_train.predictors.shape[0])) + 1
    }
    # Base model
    clf = RandomForestClassifier(random_state=54288, class_weight="balanced")
    # Instantiate the grid search model
    print('Hyperparameter optimization')
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               cv=min(4, int(mimic_train.predictors.shape[0] / 2)), n_jobs=-1, verbose=0)

    # Fit the grid search to the data
    grid_search.fit(mimic_train.predictors, mimic_train.target)

    # Get best model
    clf = grid_search.best_estimator_
    print(grid_search.best_params_)

    # Predict
    mimic_train.predict(clf)
    mimic_test1.predict(clf)
    mimic_test2.predict(clf)
    eicu_test.predict(clf)

    # Save files
    mimic_test1.save_file()
    mimic_test2.save_file()
    eicu_test.save_file()

    # Get eicu hospitalid with more than 250 patients
    hosp_id_list = eicu_test.hospitalid.value_counts()
    hosp_id_filt = hosp_id_list[hosp_id_list >= 200].index.values
    for hosp_id in hosp_id_filt:
        eicu_test.save_file(hosp_id)

    # Show metrics
    mimic_train.show_metrics()
    mimic_test1.show_metrics()
    mimic_test2.show_metrics()
    eicu_test.show_metrics()


class Dataset:
    """

    """
    __predictors = ['age', 'pao2fio2', 'uo', 'aids', 'hem', 'mets', 'admissiontype', 'bicarbonate', 'bilirubin', 'bun',
                    'gcs', 'hr', 'potassium', 'sbp', 'sodium', 'tempc', 'wbc']
    __round_predictors = ['mets', 'hem', 'aids']
    __target = 'deceased'
    __hospitaid = 'hospitalid'
    pred_probability = None
    pred_target = None
    __df = None

    def __init__(self, df: pd.DataFrame, title=''):
        self.predictors = df[self.__predictors]
        self.target = df[self.__target]
        self.hospitalid = df[self.__hospitaid]
        self.title = title

    def round(self):
        self.predictors[self.__round_predictors] = self.predictors[self.__round_predictors].round()

    def predict(self, clf):
        self.pred_probability = clf.predict_proba(self.predictors)[:, 1]
        self.pred_target = clf.predict(self.predictors)

    def get_dataframe(self, regenerate=False):
        if not regenerate and self.__df is not None:
            return self.__df

        df = self.predictors.copy(deep=True)
        df['target'] = self.target
        df['hospitalid'] = self.hospitalid
        df['pred_probability'] = self.pred_probability
        df['pred_target'] = self.pred_target
        self.__df = df
        return df

    def save_file(self, hosp_id = None):
        df = self.get_dataframe()

        if not hosp_id:
            df.to_csv(f"data/{self.title}.csv", index=False)
        else:
            df[df['hospitalid']==hosp_id].to_csv(f"data/{self.title}_{hosp_id}.csv", index=False)

    def show_metrics(self):
        print(f"\nMetrics for {self.title}")
        print(f"accuracy:{accuracy_score(self.target, self.pred_target)}")
        print(f"auc:{roc_auc_score(self.target, self.pred_probability)}")
        print(f"precision:{precision_score(self.target, self.pred_target)}")


class Imputer:
    """

    """

    def __init__(self, df: Dataset):
        self.imputer = KNNImputer(n_neighbors=20)
        df.predictors = pd.DataFrame(self.imputer.fit_transform(df.predictors), columns=df.predictors.columns)

    def impute(self, df: Dataset):
        df.predictors = pd.DataFrame(self.imputer.transform(df.predictors), columns=df.predictors.columns)
        df.round()


if __name__ == "__main__":
    testing_set_generator()

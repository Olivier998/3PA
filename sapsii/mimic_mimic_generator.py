from saps_processing import convert_saps
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("../../../data/sapsii/mimic_filtered_data.csv")
df = df.drop(columns=['stay_id', 'hospitalid'])

# get df
df_0 = df[(df['anchor_year_group'] == '2008 - 2010') | (df['anchor_year_group'] == '2011 - 2013')]
df_0 = df_0.drop(columns=['anchor_year_group'])
df_1 = df[df['anchor_year_group'] == '2014 - 2016']
df_1 = df_1.drop(columns=['anchor_year_group'])
df_2 = df[df['anchor_year_group'] == '2017 - 2019']
df_2 = df_2.drop(columns=['anchor_year_group'])

# Get outcome
y_0 = np.array(df_0['deceased'])
df_0 = df_0.drop(columns=['deceased'])
y_1 = np.array(df_1['deceased'])
df_1 = df_1.drop(columns=['deceased'])
y_2 = np.array(df_2['deceased'])
df_2 = df_2.drop(columns=['deceased'])

# Get knnimputed values
imputer = KNNImputer(n_neighbors=20)
df_0 = pd.DataFrame(imputer.fit_transform(df_0), columns=df_0.columns)
df_1 = pd.DataFrame(imputer.transform(df_1), columns=df_1.columns)
df_2 = pd.DataFrame(imputer.transform(df_2), columns=df_2.columns)

# Round variables
variables_to_round = ['mets', 'hem', 'aids', 'cpap', 'vent']
df_0[variables_to_round] = df_0[variables_to_round].round()
df_1[variables_to_round] = df_1[variables_to_round].round()
df_2[variables_to_round] = df_2[variables_to_round].round()

# Apply saps
df_0_score, _ = convert_saps(df_0)
df_1_score, _ = convert_saps(df_1)
df_2_score, _ = convert_saps(df_2)

# Train model
# Parameter grid
param_grid = {
    'max_depth': range(2, int(np.log2(df_0_score.shape[0])) + 1)
}
# Base model
clf = RandomForestClassifier(random_state=54288, class_weight="balanced")

# Instantiate the grid search model
print('Hyperparameter optimization')
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=min(4, int(df_0_score.shape[0] / 2)), n_jobs=-1, verbose=0)

# Fit the grid search to the data
grid_search.fit(df_0_score, y_0)
# print(grid_search.best_params_)

# Get best model
clf = grid_search.best_estimator_

# clf = RandomForestClassifier().fit(df_0_score, y_0)

# Predict
df_0_score['prediction'] = clf.predict_proba(df_0_score)[:, 1]
df_0_score['y_true'] = y_0

df_1_score['prediction'] = clf.predict_proba(df_1_score)[:, 1]
df_1_score['y_true'] = y_1

df_2_score['prediction'] = clf.predict_proba(df_2_score)[:, 1]
df_2_score['y_true'] = y_2

# Save results
df_0_score.to_csv('../../../data/mimic_mimic/df_0.csv', index=False)
df_1_score.to_csv('../../../data/mimic_mimic/df_1.csv', index=False)
df_2_score.to_csv('../../../data/mimic_mimic/df_2.csv', index=False)

df_12_score = pd.concat([df_1_score, df_2_score])

df_12_score.to_csv('../../../data/mimic_mimic/df_12.csv', index=False)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tree_resumer_bokeh import export_graphviz_tree
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from tqdm import tqdm


prob_str = 'probability'
y_true_str = 'oym'

data_path = '../../data/oym.csv'

df = pd.read_csv(data_path)
df['prediction'] = df[prob_str]
df['deceased'] = df[y_true_str]
df = df.drop([prob_str, y_true_str], axis=1)

df_pre = df[df['admission_year'] < 2019]
df_post = df[df['admission_year'] > 2019]

# ## To investigate only
# df_post = df_post[0:200]
# df_pre = df_pre[0:200]

# Pre section
df_train_pre = df_pre.copy()
df_train_pre = df_train_pre.drop(['prediction', 'deceased'], axis=1)
prob_pre = df_pre['prediction']
true_pre = df_pre['deceased']
error_prob_pre = 1 - np.abs(true_pre - prob_pre)

rf_pre = RandomForestRegressor(random_state=54288)
param_grid_pre = {
    'max_depth': range(2, int(np.log2(df_pre.shape[0])) + 1)
}
grid_search = GridSearchCV(estimator=rf_pre, param_grid=param_grid_pre,
                           cv=min(4, int(df_pre.shape[0] / 2)), n_jobs=-1, verbose=0)

# Fit the grid search to the data
pos_class_weight = 1 - true_pre.mean()
sample_weight = np.array([pos_class_weight if yi == 1 else 1 - pos_class_weight for yi in true_pre])
grid_search.fit(df_train_pre, error_prob_pre, sample_weight=sample_weight)
rf_pre = grid_search.best_estimator_
rf_pre_values = rf_pre.predict(df_train_pre)

tree_pre = DecisionTreeRegressor(max_depth=5, random_state=54288)
tree_pre.fit(df_train_pre, rf_pre_values)

# pre section - Theoric
df_pre_theoric = df_pre.copy()
df_pre_theoric['other_tree'] = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                         zip(rf_pre.predict(df_train_pre), tree_pre.predict(df_train_pre))])

# to predict values
df_train_pre_theoric_glob = df_pre.copy()
prob_pre_theoric_glob = df_train_pre_theoric_glob['prediction']
true_pre_theoric_glob = df_train_pre_theoric_glob['deceased']
df_train_pre_theoric_glob = df_train_pre_theoric_glob.drop(['prediction', 'deceased'], axis=1)

nb_iter = 1000
df_pre_theoric['nb_iter'] = nb_iter
for iter in tqdm(range(nb_iter)):
    df_train_pre_theoric = resample(df_pre.copy())
    prob_pre_theoric = df_train_pre_theoric['prediction']
    true_pre_theoric = df_train_pre_theoric['deceased']
    df_train_pre_theoric = df_train_pre_theoric.drop(['prediction', 'deceased'], axis=1)

    error_prob_pre_theoric = 1 - np.abs(true_pre_theoric - prob_pre_theoric)

    rf_pre_theoric = RandomForestRegressor(random_state=54288)
    param_grid_pre_theoric = {
        'max_depth': range(2, int(np.log2(df_pre.shape[0])) + 1)
    }
    grid_search = GridSearchCV(estimator=rf_pre_theoric, param_grid=param_grid_pre_theoric,
                               cv=min(4, int(df_pre.shape[0] / 2)), n_jobs=-1, verbose=0)

    # Fit the grid search to the data
    pos_class_weight = 1 - true_pre_theoric.mean()
    sample_weight = np.array([pos_class_weight if yi == 1 else 1 - pos_class_weight for yi in true_pre_theoric])
    grid_search.fit(df_train_pre_theoric, error_prob_pre_theoric, sample_weight=sample_weight)
    rf_pre_theoric = grid_search.best_estimator_
    rf_pre_theoric_values = rf_pre_theoric.predict(df_train_pre_theoric)

    tree_pre_theoric = DecisionTreeRegressor(max_depth=5, random_state=54288)
    tree_pre_theoric.fit(df_train_pre_theoric, rf_pre_theoric_values)

    # add predicted values
    df_pre_theoric[iter] = np.abs(df_pre_theoric['other_tree'] - np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                                                           zip(rf_pre_theoric.predict(
                                                                               df_train_pre_theoric_glob),
                                                                               tree_pre_theoric.predict(
                                                                                   df_train_pre_theoric_glob))]))
#  ## 95 percentile of differences
# iter_means = df_pre_theoric[range(nb_iter)].mean(axis=0)
# iter_means_idx = list(iter_means).index(iter_means.quantile(0.95, interpolation='nearest'))
#
# df_pre_theoric['diff'] = df_pre_theoric[iter_means_idx]

    #df_pre_theoric[range(nb_iter)].quantile(0.95, interpolation='nearest', axis=1)
# .mean(axis=1)  # this_tree

# Post section
df_train_post = df_post.copy()
df_train_post = df_train_post.drop(['prediction', 'deceased'], axis=1)
prob_post = df_post['prediction']
true_post = df_post['deceased']
error_prob_post = 1 - np.abs(true_post - prob_post)

rf_post = RandomForestRegressor(random_state=54288)
param_grid_post = {
    'max_depth': range(2, int(np.log2(df_post.shape[0])) + 1)
}
grid_search = GridSearchCV(estimator=rf_post, param_grid=param_grid_post,
                           cv=min(4, int(df_post.shape[0] / 2)), n_jobs=-1, verbose=0)

# Fit the grid search to the data
pos_class_weight = 1 - true_post.mean()
sample_weight = np.array([pos_class_weight if yi == 1 else 1 - pos_class_weight for yi in true_post])
grid_search.fit(df_train_post, error_prob_post, sample_weight=sample_weight)
rf_post = grid_search.best_estimator_
rf_post_values = rf_post.predict(df_train_post)

tree_post = DecisionTreeRegressor(max_depth=5, random_state=54288)
tree_post.fit(df_train_post, rf_post_values)

# predicted values
df_pre['this_tree'] = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                zip(rf_pre.predict(df_train_pre), tree_pre.predict(df_train_pre))])
df_pre['other_tree'] = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                 zip(rf_post.predict(df_train_pre), tree_post.predict(df_train_pre))])
# df_pre['diff'] = np.abs(df_pre['this_tree'] - df_pre['other_tree'])  # ** 2

# df_pre_theoric['diff'] = np.abs(df_pre_theoric['this_tree'] - df_pre_theoric['other_tree'])  # ** 2

df_post['this_tree'] = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                 zip(rf_post.predict(df_train_post), tree_post.predict(df_train_post))])
df_post['other_tree'] = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                  zip(rf_pre.predict(df_train_post), tree_pre.predict(df_train_post))])
# df_post['diff'] = np.abs(df_post['this_tree'] - df_post['other_tree'])  # ** 2

dot_data = export_graphviz_tree(tree_pre, df_pre, df_pre_theoric, df_post,
                                feature_names=df_train_pre.columns, proportion=True)

graph = graphviz.Source(dot_data)
graph.render('pretree_msqe_test_felix')

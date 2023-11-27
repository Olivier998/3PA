import pandas as pd
import numpy as np
# from tree_structure import VariableTree
from sklearn.tree import DecisionTreeRegressor
from tree_transcriber import TreeTranscriber
from tree_resumer_homr import export_graphviz_tree
import graphviz

prob_str = 'probability'
y_true_str = 'oym'

data_path = '../../data/oym.csv'

df = pd.read_csv(data_path)
df['prediction'] = df[prob_str]
df['deceased'] = df[y_true_str]
df = df.drop([prob_str, y_true_str], axis=1)

df_pre = df[df['admission_year'] < 2019]
df_post = df[df['admission_year'] > 2019]

df_train = df_pre.copy()
df_train = df_train.drop(['prediction', 'deceased'], axis=1)

prob = df_pre['prediction']

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(df_train, prob)

dot_data = export_graphviz_tree(tree, df_pre, df_post,
                                feature_names=df_train.columns, proportion=True)

graph = graphviz.Source(dot_data)
graph.render('HOMR')

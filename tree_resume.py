import pandas as pd
import numpy as np
# from tree_structure import VariableTree
from sklearn.tree import DecisionTreeRegressor
from tree_transcriber import TreeTranscriber
from tree_resumer import export_graphviz_tree
import graphviz


data_path = 'simulated_data.csv'

df = pd.read_csv(data_path)

df_base = df.copy()
df_base['prediction'] = df_base['pred_prob']
df_base['deceased'] = df_base['y_true']
df_base = df_base.drop(['y_true', 'pred_prob'], axis=1)

prob = df['pred_prob']
df = df.drop(['y_true', 'pred_prob'], axis=1)

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(df, prob)

dot_data = export_graphviz_tree(tree, df_base, df_base,
                                feature_names=df.columns, proportion=True, class_names=['0', '1'])

graph = graphviz.Source(dot_data)
graph.render('test')

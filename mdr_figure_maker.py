from bokeh.layouts import row, column, layout
from bokeh.models import Div, RangeSlider, Spinner, Slider, Arrow, NormalHead
from bokeh.plotting import figure

# Select a color palette
from bokeh.palettes import Colorblind8 as palette
import itertools

from bokeh.io import curdoc
from constants import FontSize

from bokeh.embed.standalone import file_html
from bokeh.resources import CDN

import os
import webbrowser

from sklearn.ensemble import RandomForestRegressor
from tree_structure import VariableTree
from utils import get_mdr
import numpy as np


THRESHOLD = 0.5


def generate_mdr(x, y, predicted_prob, pos_class_weight=0.5, tree_depth=2, filename=None):

    # Tool header
    tool_header = Div(text=f"<b>MDR (Positive weight = {pos_class_weight})</b>",
                      sizing_mode="stretch_width", height=100, align="center",
                      styles={"text-align": "center", "background": "grey",
                              "font-size": FontSize.TITLE})

    slider_dr = Slider(start=0, end=100, value=100, step=1, title='Declaration Rate',
                       sizing_mode="stretch_width",
                       #align='center',
                       styles={"text-align": "center", "font-size": FontSize.SUB_TITLE, "padding": "0.5vw",
                               "width": "75%", "align-self": "center"})

    # Section to train misclassification model
    error_prob = 1 - np.abs(y - predicted_prob)
    sample_weight = np.array([pos_class_weight if yi == 1 else 1 - pos_class_weight for yi in y])

    ca_rf = RandomForestRegressor()
    ca_rf.fit(x, error_prob, sample_weight=sample_weight)
    ca_rf_values = ca_rf.predict(x)

    ca_profile = VariableTree()
    ca_profile.fit(x, ca_rf_values)
    ca_profile_values = ca_profile.predict(x, depth=tree_depth)

    min_values_depth = np.array([min(rf_val, prof_val) for rf_val, prof_val in zip(ca_rf_values, ca_profile_values)])

    y_pred = np.array([1 if y_score_i >= THRESHOLD else 0 for y_score_i in predicted_prob])
    mdr_depth = get_mdr(y, y_pred, min_values_depth)

    metrics = list(mdr_depth[0].keys())
    dr = [mdr_depth[i]['dr'] for i in range(len(mdr_depth))]
    metrics.remove('dr')

    # color manager
    colors = itertools.cycle(palette)

    # Plot metrics
    plot_metrics = figure(x_axis_label='Declaration Rate', y_axis_label='Metrics score')
    for metric_name, color in zip(metrics, colors):
        plot_metrics.line(dr, [mdr_depth[i][metric_name] for i in range(len(mdr_depth))],
                          legend_label=metric_name, line_width=2, color=color)

        #ax.plot(dr, [mdr_depth[i][metric_name] for i in range(len(mdr_depth))], label=metric_name, linewidth=2)

    # Setup legend
    plot_metrics.legend.click_policy = "hide"
    plot_metrics.right = plot_metrics.legend

    # Plot tree
    plot_tree = figure()
    plot_tree.axis.visible = False
    plot_tree.grid.visible = False

    # First node
    plot_tree.rect(x=0, y=0, width=20, height=10, fill_color='white', line_color='black', line_width=2)

    # second layer
    plot_tree.rect(x=-20, y=-15, width=20, height=10, fill_color='white', line_color='black', line_width=2)
    plot_tree.rect(x=20, y=-15, width=20, height=10, fill_color='white', line_color='black', line_width=2)

    # Arrows of second layer
    plot_tree.add_layout(Arrow(x_start=0, y_start=-5, x_end=-20, y_end=-10, end=NormalHead()))
    plot_tree.add_layout(Arrow(x_start=0, y_start=-5, x_end=20, y_end=-10, end=NormalHead()))

    #tt1 = Div(text="Box1")
    #tt2 = Div(text="Box2")

    # We set the two box (extracted profiles and MDR curves)
    outline_boxs = row(
        row(
            column(
                plot_tree,
                sizing_mode="stretch_both",
                styles={"text-align": "center",
                        "font-size": FontSize.NORMAL,
                        "padding": "1vw",
                        "border": "1px solid black"}),
            sizing_mode="stretch_both",
            styles={"padding": "0.5vw"}),
        row(
            column(
                plot_metrics,
                sizing_mode="stretch_both",
                styles={"text-align": "center",
                        "font-size": FontSize.NORMAL,
                        "padding": "1vw",
                        "border": "1px solid black"}),
            sizing_mode="stretch_both",
            styles={"padding": "0.5vw"}
        ),
        sizing_mode="stretch_both"
    )

    # create layout
    layout_output = layout([
        tool_header,
        slider_dr,
        [outline_boxs],
    ],
        sizing_mode="stretch_both")

    # show result
    curr_doc = curdoc()  # Document()
    curr_doc.add_root(layout_output)  # (row(inputs, plot, width=1200))

    if filename is None or filename == "":
        filename = 'newtest01'
    #    output_file(filename)
    #show(layout_output)
    html = file_html(layout_output, CDN, title=filename)

    path = os.path.abspath(filename + '.html')

    with open(path, 'w') as file:
        file.write(html)

    webbrowser.open(url=path)


if __name__ == '__main__':
    # prepare some data
    import pandas as pd
    import numpy as np
    x = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]], columns=['a','b','c'])  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.array([1, 1, 0, 0])
    y_pred = np.array([0.98, 0.45, 0.35, 0.02])
    generate_mdr(x, y, y_pred)

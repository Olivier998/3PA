from bokeh.layouts import row, column, layout
from bokeh.models import Div, RangeSlider, Spinner, Slider, LabelSet, ColumnDataSource, CustomJSFilter, CDSView, \
    IndexFilter, Line
from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS

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
from tree_transcriber import TreeTranscriber
from utils import get_mdr
import numpy as np

THRESHOLD = 0.5

# Metrics names
AUC = 'auc'
BAL_ACC = 'bal_acc'
MEAN_CA = 'mean_ca'
PERC_POP = 'perc_pop'
PERC_NODE = 'perc_node'
SENSITIVITY = 'sens'
SPECIFICITY = 'spec'

METRICS = [BAL_ACC, SENSITIVITY, SPECIFICITY, AUC, MEAN_CA, PERC_POP, PERC_NODE]
METRICS_MDR = [BAL_ACC, SENSITIVITY, SPECIFICITY, AUC]


def generate_mdr(x, y, predicted_prob, pos_class_weight=0.5, tree_depth=2, filename=None):
    # Tool header
    tool_header = Div(text=f"<b>MDR (Positive weight = {pos_class_weight})</b>",
                      sizing_mode="stretch_width", align="center",  # height="5vw" ,
                      styles={"text-align": "center", "background": "grey",
                              "font-size": FontSize.TITLE},
                      stylesheets=[":host {height: 5vw;}"])

    # Sliders
    slider_dr = Slider(start=0, end=100, value=100, step=1, title='Declaration Rate',
                       sizing_mode="stretch_width",
                       styles={"text-align": "center", "font-size": FontSize.SUB_TITLE, "padding": "0.5vw",
                               "width": "75%", "align-self": "center"})
    slider_depth = Slider(start=0, end=4, value=4, step=1, title='Tree depth',
                          sizing_mode="stretch_width",
                          styles={"text-align": "center", "font-size": FontSize.SUB_TITLE, "padding": "0.5vw",
                                  "width": "75%", "align-self": "center"})

    # Section to train misclassification model
    y_pred = np.array([1 if y_score_i >= THRESHOLD else 0 for y_score_i in predicted_prob])
    error_prob = 1 - np.abs(y - predicted_prob)
    sample_weight = np.array([pos_class_weight if yi == 1 else 1 - pos_class_weight for yi in y])

    ca_rf = RandomForestRegressor()
    ca_rf.fit(x, error_prob, sample_weight=sample_weight)
    ca_rf_values = ca_rf.predict(x)

    ca_profile = VariableTree(max_depth=slider_depth.end)
    ca_profile.fit(x, ca_rf_values)

    min_cas = {}
    mdr_depths_dict = {'depth': [], 'values': []}
    for depth in range(0, slider_depth.end + 1):
        if depth == 0:
            min_values_depth = ca_rf_values
        else:
            ca_profile_values = ca_profile.predict(x, depth=depth)
            min_values_depth = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                         zip(ca_rf_values, ca_profile_values)])
        min_cas[depth] = min_values_depth

        # Get mdr values for every depth
        mdr_values = get_mdr(y, y_pred, min_values_depth)
        # from list of dicts to dict
        mdr_dict = {k: [dic[k] for dic in mdr_values] for k in mdr_values[0]}

        # Save values
        mdr_depths_dict['depth'].append(depth)
        mdr_depths_dict['values'].append(mdr_dict)

    mdr_depths_data = ColumnDataSource(data=mdr_depths_dict)
    index_current_data = mdr_depths_dict['depth'].index(slider_depth.value)
    mdr_current_data = ColumnDataSource(data=mdr_depths_dict['values'][index_current_data])

    # color manager
    colors = itertools.cycle(palette)

    # Plot metrics
    plot_metrics = figure(x_axis_label='Declaration Rate', y_axis_label='Metrics score')
    plot_metrics.axis.axis_label_text_font_style = 'bold'
    for metric_name, color in zip(METRICS_MDR, colors):
        plot_metrics.line(x='dr', y=metric_name,
                          legend_label=metric_name, line_width=2, color=color, source=mdr_current_data)

    # Setup legend
    plot_metrics.legend.click_policy = "hide"
    plot_metrics.right = plot_metrics.legend

    # Plot tree
    plot_tree = figure(aspect_ratio=1, aspect_scale=1, match_aspect=True)  # tools=WheelZoomTool())
    plot_tree.axis.visible = False
    plot_tree.grid.visible = False

    # Get tree nodes
    tree_getter = TreeTranscriber(tree=ca_profile, dimensions=[20, 16], min_ratio_leafs=0.5, metrics=METRICS)
    nodes, arrows, nodes_text = tree_getter.render_to_bokeh(x=x, y_true=y, y_prob=predicted_prob, min_cas=min_cas,
                                                            depth=slider_depth.value)
    for node in nodes:
        plot_tree.add_glyph(node)

    for arrow in arrows:
        plot_tree.add_layout(arrow)

    # from list of dict to dict
    nodes_text_dict = {k: [dic[k] for dic in nodes_text] for k in nodes_text[0]}

    nodes_labels = ColumnDataSource(data=nodes_text_dict)
    nodes_labelset = LabelSet(x='x', y='y', text='text', source=nodes_labels)
    plot_tree.add_layout(nodes_labelset)

    # Set nodes text values
    for text_id in range(len(nodes_labels.data['x'])):
        if nodes_labels.data['curr_depth'][text_id] <= slider_depth.value:
            node_id = nodes_labels.data['node_id'][text_id]
            curr_metric = nodes_labels.data['metric'][text_id]
            for node in nodes:
                if node.tags[0]['node_id'] == node_id:
                    id_depth = node.tags[1]['depth'].index(slider_depth.value)
                    node_values = node.tags[1]['values'][id_depth]
                    dr_array = np.array(node_values['dr'])
                    id_min_dr = dr_array[dr_array <= slider_dr.value / 100].argmax()
                    metric_value = node_values['metrics'][id_min_dr][curr_metric]
                    nodes_labels.data['text'][text_id] = f'{curr_metric} = {metric_value}'
                    break

    cjs = CustomJS(args=dict(labels=[nodes_labelset], width=20, figure=plot_tree), code="""
    var ratio = (4 * width / (figure.x_range.end-figure.x_range.start));
    for (let i = 0; i < labels.length; i++){
        labels[i].text_font_size = ratio+'vw';
    }
    """)
    plot_tree.x_range.js_on_change('start', cjs)

    str_update_mdr = """
    // Change the MDR section
    var data=src.data;
    var depths=data['depth'];
    var values=data['values'];
    
    var depth_index=depths.indexOf(slider_depth.value);
    
    curr.data=values[depth_index];
    curr.change.emit();
    """

    str_update_profile = """
    function find_smallest_id(arr, value)
    {  // Returns the id of the closest item in arr that is <= than value
        var n = arr.length;
        var smallest_id = -1;
        var smallest_value = -Infinity;
        for (var i = 0; i < n; i++)
        {
            if(smallest_value < arr[i] && arr[i] <= value)
            {
                smallest_id = i;
                smallest_value = arr[i];
            }
        }
        return smallest_id;
    }                          
                  
    var depth=slider_depth.value;
    
    // Change the Profile section
    for (var i=0; i< nodes.length; i++)
    {
        if (nodes[i].tags[0]['curr_depth'] > depth)
        {
            var remove_node = true;
        }
        else
        {
            var id_depth = nodes[i].tags[1]['depth'].indexOf(depth);
            var node_values = nodes[i].tags[1]['values'][id_depth];
            var id_min_dr = find_smallest_id(node_values['dr'], slider_dr.value/100);
            
            if (id_min_dr == -1)
            {
                var remove_node = true;
            }
            else
            {
                var remove_node = false;
            }
        }
        if (remove_node)
        {  // Remove text of the node
            nodes[i].line_alpha = 0;
            for (var j=0; j< labels.data['text'].length; j++)
            {
                if (nodes[i].tags[0]['node_id'] == labels.data['node_id'][j])
                {
                    labels.data['text'][j] = '';
                }
            }
        }
        else 
        {
            nodes[i].line_alpha = 1;
            for (var j=0; j< labels.data['text'].length; j++)
            {
                if (nodes[i].tags[0]['node_id'] == labels.data['node_id'][j])
                {
                    var met_name = labels.data['metric'][j]
                    var met_value = node_values['metrics'][id_min_dr][met_name]
                    labels.data['text'][j] = met_name + ' = ' + met_value;
                    nodes[i].line_alpha = 1;
                }
            }
        }
    }
    labels.change.emit();
    """

    callback_dr = CustomJS(args=dict(slider_depth=slider_depth, slider_dr=slider_dr, nodes=nodes, labels=nodes_labels),
                           code=str_update_profile)
    callback_depth = CustomJS(args=dict(src=mdr_depths_data, curr=mdr_current_data, slider_depth=slider_depth,
                                        slider_dr=slider_dr, nodes=nodes, labels=nodes_labels),
                              code=str_update_profile + str_update_mdr)

    # set callback actions
    slider_depth.js_on_change('value', callback_depth)
    slider_dr.js_on_change('value', callback_dr)

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
        row(slider_dr,
            slider_depth, sizing_mode="stretch_width"),
        [outline_boxs],
    ],
        sizing_mode="stretch_both")

    # show result
    curr_doc = curdoc()  # Document()
    curr_doc.add_root(layout_output)  # (row(inputs, plot, width=1200))

    if filename is None or filename == "":
        filename = 'newtest01'
    html = file_html(layout_output, CDN, title=filename)

    path = os.path.abspath(filename + '.html')

    with open(path, 'w') as file:
        file.write(html)

    webbrowser.open(url=path)


if __name__ == '__main__':
    # prepare some data
    import pandas as pd

    x = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                     columns=['a', 'b', 'c'])  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.array([1, 1, 0, 0])
    y_pred = np.array([0.98, 0.45, 0.35, 0.02])
    generate_mdr(x, y, y_pred)

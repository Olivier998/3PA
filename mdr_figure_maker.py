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
METRICS = ['bal_acc', 'sens', 'spec']


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
    for metric_name, color in zip(METRICS, colors):
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
    tree_getter = TreeTranscriber(tree=ca_profile, dimensions=[20, 10], min_ratio_leafs=0.5)
    nodes, arrows = tree_getter.render_to_bokeh(x=x, y_true=y, y_prob=predicted_prob, min_cas=min_cas, depth=2)
    for node in nodes:
        plot_tree.add_glyph(node)

    for arrow in arrows:
        plot_tree.add_layout(arrow)

    # test
    labels = ColumnDataSource(data=dict(x=[-9, -9, -9],
                                        y=[3, 1, -1],
                                        label=[0, 1, 0],
                                        values=['salut', 'auc = 0.5', 'spec=9']))
    labels2 = ColumnDataSource(data=dict(x=[-9],
                                         y=[3],
                                         label=[0],
                                         values=['salut']))

    active = list(range(3))
    filter = IndexFilter(indices=active)

    labelset = LabelSet(x='x', y='y', text='values', source=labels2)
    plot_tree.add_layout(labelset)  # , view=view

    callback = CustomJS(args=dict(src=labels, lsrc=labels2, filt=filter), code='''
    filt.indices = [cb_obj.value];//[...Array(cb_obj.value).keys()];
    src.change.emit();

    var data=src.data;
    var l=data['label'];
    var x=data['x'];
    var y=data['y'];
    var values=data['values'];

    var ldata=lsrc.data;
    var a=[cb_obj.value];//[...Array(cb_obj.value).keys()];

    var ll=[];
    var lv=[];
    var lx=[];
    var ly=[];
    for(var i=0;i<a.length;i++){
        ll.push(l[a[i]]);
        lx.push(x[a[i]]);
        ly.push(y[a[i]]);
        lv.push(values[a[i]]);
    }

    ldata['label']=ll;
    ldata['x']=lx;
    ldata['y']=ly;
    ldata['values']=lv;
    lsrc.change.emit();
    ''')

    cjs = CustomJS(args=dict(labels=[labelset], width=20, figure=plot_tree), code="""
    var ratio = (6 * width / (figure.x_range.end-figure.x_range.start));
    for (let i = 0; i < labels.length; i++){
        labels[i].text_font_size = ratio+'vw';
    }
    """)
    plot_tree.x_range.js_on_change('start', cjs)

    callback2 = CustomJS(args=dict(source=labels), code="""
        source.change.emit();
    """)
    slider_dr.js_on_change('value', callback)

    callback_depth = CustomJS(args=dict(src=mdr_depths_data, curr=mdr_current_data, slider=slider_depth, nodes=nodes),
                              code='''
    var depth=slider.value;
    
    // Change the MDR section
    var data=src.data;
    var depths=data['depth'];
    var values=data['values'];
    
    var depth_index=depths.indexOf(depth);
    
    curr.data=values[depth_index];
    curr.change.emit();
    
    // Change the Profile section
    console.log(nodes[0].tags[1]);
    for (var i=0; i< nodes.length; i++){
        if (nodes[i].tags[0] > depth){
            nodes[i].line_alpha = 0;
        }
        else{
            nodes[i].line_alpha=1;
        }
    }
    
    ''')
    slider_depth.js_on_change('value', callback_depth)




    # js_filter = CustomJSFilter(args=dict(slider=slider_dr), code="""  # , source=labels
    #    if (slider.value >50){
    #        desiredElementCount=3;
    #    }
    #    else {
    #    desiredElementCount =2;
    #    }#
    #
    #        return [...Array(desiredElementCount).keys()];
    #   """)

    #  view = CDSView(filter=[js_filter])

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
    import numpy as np

    x = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                     columns=['a', 'b', 'c'])  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.array([1, 1, 0, 0])
    y_pred = np.array([0.98, 0.45, 0.35, 0.02])
    generate_mdr(x, y, y_pred)

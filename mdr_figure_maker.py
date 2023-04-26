from bokeh.layouts import row, column, layout
from bokeh.models import Div, Slider, LabelSet, ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, SaveTool, \
    PanTool, Button, TapTool
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
from sklearn.model_selection import GridSearchCV
from tree_structure import VariableTree
from tree_transcriber import TreeTranscriber
from utils import get_mdr
import numpy as np
import time

THRESHOLD = 0.5

# Metrics names
AUC = 'auc'
AUPRC = 'auprc'
ACC = 'accuracy'
BAL_ACC = 'bal_acc'
MCC = 'mcc'
MEAN_CA = 'mean_ca'
NPV = 'npv'
PERC_POP = 'perc_pop'
PERC_NODE = 'perc_node'
PERC_POS = 'pos_perc'
PPV = 'ppv'
SENSITIVITY = 'sens'
SPECIFICITY = 'spec'
DR = 'dr'

METRICS_DISPLAY = {PERC_POS: '% positive', AUC: 'Auc', AUPRC: 'Auprc', BAL_ACC: 'Bal_Acc', MEAN_CA: 'Mean CA',
                   PERC_POP: '% pop', PERC_NODE: '% node', SENSITIVITY: 'sens',
                   SPECIFICITY: 'spec', DR: 'DR', ACC: 'Acc', MCC: 'Mcc', PPV: 'PPV', NPV: 'NPV'}

METRICS = [PERC_POS, BAL_ACC, SENSITIVITY, SPECIFICITY, AUC, MEAN_CA, PERC_POP, PERC_NODE]
METRICS_MDR = [METRICS_DISPLAY[metric] for metric in [BAL_ACC, SENSITIVITY, SPECIFICITY, AUC, AUPRC, MCC, PPV, NPV]]


def generate_mdr(x, y, predicted_prob, pos_class_weight=0.5, filename=None):
    if filename is None or filename == "":
        filename = 'newtest01'
    curr_time = int(time.time())
    print('Starting MDR process')
    # Tool header
    tool_header = Div(text=f"<b> MDR </b> <small><small>(Positive weight = {pos_class_weight})</small></small>",
                      sizing_mode="stretch_width", align="center",  # height="5vw" ,
                      styles={"text-align": "center", "background": "grey",
                              "font-size": FontSize.TITLE},
                      stylesheets=[":host {height: 5vw;}"])

    # Sliders
    slider_dr = Slider(start=0, end=100, value=100, step=1, title='Declaration Rate',
                       sizing_mode="stretch_width",
                       styles={"text-align": "center", "font-size": FontSize.SUB_TITLE, "padding": "0.5vw",
                               "width": "75%", "align-self": "center"})
    slider_minleaf = Slider(start=0, end=45, value=0, step=5, title='Min sample % in leafs',
                            sizing_mode="stretch_width",
                            styles={"text-align": "center", "font-size": FontSize.SUB_TITLE, "padding": "0.5vw",
                                    "width": "75%", "align-self": "center"})
    max_depth_log = int(np.log2(x.shape[0]))
    max_depth_profile = min(max_depth_log, 5)
    slider_maxdepth = Slider(start=1, end=max_depth_profile, value=max_depth_profile, step=1, title='Max depth',
                             sizing_mode="stretch_width",
                             styles={"text-align": "center", "font-size": FontSize.SUB_TITLE, "padding": "0.5vw",
                                     "width": "75%", "align-self": "center"})
    bttn_save_profiles = Button(label='Save Profiles',
                                align='center',
                                #styles={'width': '15vw', 'height': '3vw'},
                                stylesheets=[".bk-btn-default {font-size: 1vw; font-weight: bold;}"])

    # Section to train misclassification model
    y_pred = np.array([1 if y_score_i >= THRESHOLD else 0 for y_score_i in predicted_prob])
    error_prob = 1 - np.abs(y - predicted_prob)
    sample_weight = np.array([pos_class_weight if yi == 1 else 1 - pos_class_weight for yi in y])

    # Parameter grid
    param_grid = {
        'max_depth': range(2, max_depth_log + 1)
    }

    # Base model
    ca_rf = RandomForestRegressor(random_state=54288)

    # Instantiate the grid search model
    print('Hyperparameter optimization')
    grid_search = GridSearchCV(estimator=ca_rf, param_grid=param_grid,
                               cv=min(4, int(x.shape[0] / 2)), n_jobs=-1, verbose=0)

    # Fit the grid search to the data
    grid_search.fit(x, error_prob, sample_weight=sample_weight)
    print(grid_search.best_params_)
    print(f"HP done: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    # Get best model
    ca_rf = grid_search.best_estimator_

    # ca_rf.fit(x, error_prob, sample_weight=sample_weight)
    ca_rf_values = ca_rf.predict(x)
    print(f"CA RF: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    ca_profile = VariableTree(max_depth=max_depth_profile, min_sample_ratio=slider_minleaf.start)
    ca_profile.fit(x, ca_rf_values)
    print(f"CA PROFILE: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    min_cas = {}
    mdr_sampratio_dict = {'samp_ratio': [], 'values': []}
    for min_perc in range(slider_minleaf.start,
                          slider_minleaf.end + slider_minleaf.step,
                          slider_minleaf.step):
        ca_profile_values = ca_profile.predict(x, min_samples_ratio=min_perc)
        min_values_sampratio = np.array([min(rf_val, prof_val) for rf_val, prof_val in
                                         zip(ca_rf_values, ca_profile_values)])
        min_cas[min_perc] = min_values_sampratio

        #temp
        if min_perc==0:
            temp = x.copy()
            temp['carf'] = ca_rf_values
            temp['cadt'] = ca_profile_values
            temp['min'] = min_values_sampratio
            #temp.to_csv('test.csv')

        # Get mdr values for every samples ratio
        mdr_values = get_mdr(y, y_pred, predicted_prob, min_values_sampratio)
        # from list of dicts to dict of lists
        mdr_dict = {METRICS_DISPLAY[k]: [dic[k] for dic in mdr_values] for k in mdr_values[0]}

        dr_profile = []
        dr_lost_profiles = []
        dr_lost_profiles_id = []
        sorted_accuracies = np.sort(min_values_sampratio)

        dr_range = []
        prev_min_acc = -1
        for dr in range(100, 0, -1):
            curr_min_acc = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr/100))]
            if prev_min_acc < curr_min_acc:
                prev_min_acc = curr_min_acc
                dr_range.append([dr, curr_min_acc])
        dr_range.append([0, 1.01])

        profiles_curr, profiles_curr_id = ca_profile.get_all_profiles(min_ca=dr_range[0][1], min_samples_ratio=min_perc)

        for id in range(len(dr_range)-1):
            dr, min_ca_curr = dr_range[id]
            _, min_ca_next = dr_range[id+1]

            profiles_next, profiles_next_id = ca_profile.get_all_profiles(min_ca=min_ca_next,
                                                                          min_samples_ratio=min_perc)
            # print(f"{min_ca_curr} {min_ca_next} {len(profiles_curr)}  {len(profiles_next)} {dr}  {len(min_values_sampratio[min_values_sampratio >= min_ca_curr]) /len(ca_profile_values)}\n")
            if len(profiles_curr) != len(profiles_next):
                lost_profiles = list(set(profiles_curr) - set(profiles_next))
                lost_profiles_id = list(set(profiles_curr_id) - set(profiles_next_id))
                # print(f"{lost_profiles=}")
                dr_profile.append(int(100 * len(min_values_sampratio[min_values_sampratio >= min_ca_curr]) /
                                      len(ca_profile_values)))

                dr_lost_profiles.append("<br>".join(lost_profiles))
                dr_lost_profiles_id.append((lost_profiles_id))
            profiles_curr = profiles_next
            profiles_curr_id = profiles_next_id
        #if min_perc==0:
        #    print(f"{unique_ca_profile_values[len(unique_ca_profile_values)-2]}  "
        #          f"{unique_ca_profile_values[len(unique_ca_profile_values)-1]}")
        """for min_ca_id in range(0, len(unique_ca_profile_values)-1):
            min_ca_curr = unique_ca_profile_values[min_ca_id]
            min_ca_next = unique_ca_profile_values[min_ca_id + 1]

            profiles_curr = ca_profile.get_all_profiles(min_ca=min_ca_curr, min_samples_ratio=min_perc)
            profiles_next = ca_profile.get_all_profiles(min_ca=min_ca_next, min_samples_ratio=min_perc)

            print(f"{min_ca_id=} {min_ca_curr} {min_ca_next} {len(profiles_curr)}  {len(profiles_next)}  {len(min_values_sampratio[min_values_sampratio >= min_ca_curr]) /len(ca_profile_values)}")

            if len(profiles_curr) != len(profiles_next):
                lost_profiles = list(set(profiles_curr) - set(profiles_next))
                dr_profile.append(int(100 * len(min_values_sampratio[min_values_sampratio >= min_ca_curr]) /
                                      len(ca_profile_values)))

                dr_lost_profiles.append("<br>".join(lost_profiles))
        """
        # dr_profile = [int(100 * len(min_values_sampratio[min_values_sampratio >= min_ca]) / len(ca_profile_values))
        #              for min_ca in unique_ca_profile_values]
        # dr_profile_y = [len(min_values_sampratio[min_values_sampratio >= min_ca]) / len(ca_profile_values)
        #                for min_ca in unique_ca_profile_values]
        # print(f"{dr_profile=}")
        dr_profile_x = [dr if dr in dr_profile else np.nan for dr in
                        mdr_dict[METRICS_DISPLAY[DR]]]  # 100 if dr == 100 else
        dr_profile_y = [0 if dr in dr_profile else np.nan for dr in  # dr / 100
                        mdr_dict[METRICS_DISPLAY[DR]]]
        dr_profile_lost = [dr_lost_profiles[dr_profile.index(dr)] if dr in dr_profile else np.nan for dr in  # dr / 100
                           mdr_dict[METRICS_DISPLAY[DR]]]
        dr_profile_lost_id = [dr_lost_profiles_id[dr_profile.index(dr)] if dr in dr_profile else np.nan for dr in
                              mdr_dict[METRICS_DISPLAY[DR]]]

        mdr_dict['dr_profile_x'] = dr_profile_x
        mdr_dict['dr_profile_y'] = dr_profile_y
        mdr_dict['dr_profile_y_line'] = [0] * len(mdr_dict[METRICS_DISPLAY[DR]])
        mdr_dict['dr_profile_lost'] = dr_profile_lost
        mdr_dict['dr_profile_lost_id'] = dr_profile_lost_id
        # [0 for dr in mdr_dict[METRICS_DISPLAY[DR]]]  # dr / 100
        #if min_perc == 0:
        #    print(f"{unique_ca_profile_values=}")
        #    print(100 * len(min_values_sampratio[min_values_sampratio >= unique_ca_profile_values[-2]]) /
        #          len(ca_profile_values))
        # Save values
        mdr_sampratio_dict['samp_ratio'].append(min_perc)
        mdr_sampratio_dict['values'].append(mdr_dict)

        print(f"get mdr {min_perc}: {int(time.time() - curr_time)}s")
        curr_time = time.time()

    mdr_sampratio_data = ColumnDataSource(data=mdr_sampratio_dict)
    index_current_data = mdr_sampratio_dict['samp_ratio'].index(slider_minleaf.value)
    mdr_current_data = ColumnDataSource(data=mdr_sampratio_dict['values'][index_current_data])

    # color manager
    colors = itertools.cycle(palette)

    # Plot metrics
    mdr_hover = HoverTool(tooltips=[('Declaration rate', f'@{METRICS_DISPLAY[DR]}'),
                                    (BAL_ACC, f'@{METRICS_DISPLAY[BAL_ACC]}'),
                                    (SENSITIVITY, f'@{METRICS_DISPLAY[SENSITIVITY]}'),
                                    (SPECIFICITY, f'@{METRICS_DISPLAY[SPECIFICITY]}'),
                                    (AUC, f'@{METRICS_DISPLAY[AUC]}'),
                                    (AUPRC, f'@{METRICS_DISPLAY[AUPRC]}'),
                                    (MCC, f'@{METRICS_DISPLAY[MCC]}'),
                                    (PPV, f'@{METRICS_DISPLAY[PPV]}'),
                                    (NPV, f'@{METRICS_DISPLAY[NPV]}'),
                                    ])
    profile_hover = HoverTool(tooltips=[('Declaration rate', '@dr_profile_x'),
                                        ('Profiles', '@dr_profile_lost{safe}'),
                                        ])

    profile_tap = TapTool(behavior='select')
    mdr_tools = [PanTool(), WheelZoomTool(), SaveTool(), ResetTool(), mdr_hover]

    plot_metrics = figure(y_axis_label='Metrics score', sizing_mode='scale_width',
                          y_range=(0.45, 1.05), tools=mdr_tools)
    plot_metrics.axis.axis_label_text_font_style = 'bold'
    for metric_name, color in zip(METRICS_MDR, colors):
        plot_metrics.line(x=METRICS_DISPLAY[DR], y=metric_name,
                          legend_label=metric_name, line_width=2, color=color, source=mdr_current_data)
        plot_metrics.circle(x=METRICS_DISPLAY[DR], y=metric_name, legend_label=metric_name,
                            line_width=2, color=color, source=mdr_current_data)

    # plot_metrics.line(x=METRICS_DISPLAY[DR], y='dr_profile_y_line', legend_label='Declaration Rate', color='black',
    #                  line_width=1, source=mdr_current_data)
    # plot_metrics.triangle_dot(x='dr_profile_x', y='dr_profile_y', legend_label='Declaration Rate', color='black',
    #                          line_width=3, source=mdr_current_data)

    plot_metrics_dr = figure(x_axis_label='Declaration Rate', sizing_mode='scale_width',  # height="2vh",
                             y_range=(-0.1, 0.9),
                             tools=[profile_hover, profile_tap],
                             stylesheets=[":host {height: 25vh;}"])

    #plot_metrics_dr.line(x='dr_profile_x', y='dr_profile_y_line', legend_label='Declaration Rate', color='black',
    #                     line_width=1, source=mdr_current_data)
    plot_metrics_dr.triangle_dot(x='dr_profile_x', y='dr_profile_y', legend_label='Declaration Rate', color='black',
                                 line_width=3, source=mdr_current_data)
    plot_metrics_dr.axis.axis_label_text_font_style = 'bold'
    plot_metrics_dr.legend.click_policy = "hide"
    plot_metrics_dr.legend.label_text_font_size = '0.75vw'
    plot_metrics_dr.right = plot_metrics_dr.legend
    plot_metrics_dr.yaxis.visible = False
    plot_metrics_dr.ygrid.visible = False

    # Setup legend
    plot_metrics.legend.click_policy = "hide"
    plot_metrics.legend.label_text_font_size = '0.75vw'
    plot_metrics.right = plot_metrics.legend

    print(f"PLOT MDR: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    # Plot tree
    plot_tree = figure(aspect_ratio=1, aspect_scale=1, match_aspect=True,
                       sizing_mode='scale_width')  # tools=WheelZoomTool())
    plot_tree.axis.visible = False
    plot_tree.grid.visible = False

    # Get tree nodes
    tree_getter = TreeTranscriber(tree=ca_profile, dimensions=[20, 18], min_ratio_leafs=0., metrics=METRICS)
    nodes, arrows, nodes_text = tree_getter.render_to_bokeh(x=x, y_true=y, y_prob=predicted_prob, min_cas=min_cas)
    print(f"Get tree values: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    for node in nodes:
        plot_tree.add_glyph(node)

    for arrow in arrows:
        plot_tree.add_layout(arrow)

    # from list of dict to dict
    nodes_text_dict = {k: [dic[k] for dic in nodes_text] for k in nodes_text[0]}

    nodes_labels = ColumnDataSource(data=nodes_text_dict)
    nodes_labelset = LabelSet(x='x', y='y', text='text', text_font_style='text_font_style', source=nodes_labels)
    plot_tree.add_layout(nodes_labelset)

    print(f"PLOT TREE: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    # Set nodes text values
    for node in nodes:
        if node.tags[0]['samp_ratio'] < slider_minleaf.value:
            remove_node = True
        else:
            id_samp_ratio = node.tags[1]['samp_ratio'].index(slider_minleaf.value)
            node_values = node.tags[1]['values'][id_samp_ratio]
            dr_array = np.array(node_values['dr'])
            id_min_dr = -1
            id_min_val = -1
            for dr_id, dr_val in enumerate(dr_array):
                if (dr_val <= slider_dr.value / 100) and (dr_val > id_min_val):
                    id_min_val = dr_val
                    id_min_dr = dr_id

            # id_min_dr = dr_array[dr_array <= slider_dr.value / 100].argmax() if \
            #    len(dr_array[dr_array <= slider_dr.value / 100]) > 0 else -1

            if id_min_dr == -1:
                remove_node = True
            else:
                remove_node = False
        if remove_node:
            node.line_alpha = 0.25
            for arrow in arrows:
                if arrow.tags[0]['node_id'] == node.tags[0]['node_id']:
                    arrow.line_alpha = 0.25
                    break
        else:
            for text_id in range(len(nodes_labels.data['x'])):
                if node.tags[0]['node_id'] == nodes_labels.data['node_id'][text_id]:
                    curr_metric = nodes_labels.data['metric'][text_id]
                    if curr_metric == 'split':
                        nodes_labels.data['text'][text_id] = node.tags[0]['split']
                    else:
                        metric_display = METRICS_DISPLAY[curr_metric]
                        metric_value = node_values['metrics'][id_min_dr][curr_metric]
                        nodes_labels.data['text'][text_id] = f'{metric_display} = {metric_value}'

    print(f"PLOT NODES: {int(time.time() - curr_time)}s")
    curr_time = time.time()

    cjs = CustomJS(args=dict(labels=[nodes_labelset], width=20, figure=plot_tree), code="""
    var ratio = (4 * width / (figure.x_range.end-figure.x_range.start));
    for (let i = 0; i < labels.length; i++){
        labels[i].text_font_size = ratio+'vw';
    }
    """)
    plot_tree.x_range.js_on_change('start', cjs)
    plot_tree.x_range.js_on_change('end', cjs)

    # Update DR profiles range when moving in the MDR figure
    mdr_fig_dr_move = CustomJS(args=dict(figure_mdr=plot_metrics, figure_dr=plot_metrics_dr), code="""
    figure_dr.x_range = figure_mdr.x_range;
    """)
    plot_metrics.x_range.js_on_change('start', mdr_fig_dr_move)
    plot_metrics.x_range.js_on_change('end', mdr_fig_dr_move)

    str_update_mdr = """
    // Change the MDR section
    var data=src.data;
    var samp_ratios=data['samp_ratio'];
    var values=data['values'];
    
    var samp_ratio_index=samp_ratios.indexOf(slider_samp_ratio.value);
    
    curr.data=values[samp_ratio_index];
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
                  
    var samp_ratio=slider_samp_ratio.value;
    var max_depth = slider_maxdepth.value;
    
    // Change the Profile section
    for (var i=0; i< nodes.length; i++)
    {
        if (nodes[i].tags[0]['samp_ratio'] < samp_ratio || nodes[i].tags[0]['curr_depth'] > max_depth)
        {
            var remove_node = true;
        }
        else
        {
            var id_samp_ratio = nodes[i].tags[1]['samp_ratio'].indexOf(samp_ratio);
            var node_values = nodes[i].tags[1]['values'][id_samp_ratio];
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
        for (var arr_id=0; arr_id < arrows.length; arr_id++)
        {
            if (arrows[arr_id].tags[0]['node_id'] == nodes[i].tags[0]['node_id'])
            {
                if (remove_node)
                {
                    arrows[arr_id].line_alpha = 0.25;
                }
                else
                {
                    arrows[arr_id].line_alpha = 1;
                }
            }
        }
        
        if (remove_node)
        {  // Remove text of the node
            nodes[i].line_alpha = 0.25;
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
                    var met_name = labels.data['metric'][j];
                    if (met_name == 'split')
                    {
                        labels.data['text'][j] = nodes[i].tags[0]['split'];
                    }
                    else
                    {
                        var met_display = metrics_display[met_name];
                        var met_value = node_values['metrics'][id_min_dr][met_name];
                        labels.data['text'][j] = met_display + ' = ' + met_value;
                    }
                }
            }
        }
    }
    labels.change.emit();
    """

    callback_dr = CustomJS(args=dict(slider_samp_ratio=slider_minleaf, slider_dr=slider_dr,
                                     slider_maxdepth=slider_maxdepth,
                                     nodes=nodes, labels=nodes_labels, arrows=arrows,
                                     metrics_display=METRICS_DISPLAY),
                           code=str_update_profile)
    callback_samp_ratio = CustomJS(args=dict(src=mdr_sampratio_data, curr=mdr_current_data,
                                             slider_maxdepth=slider_maxdepth,
                                             slider_samp_ratio=slider_minleaf, slider_dr=slider_dr,
                                             nodes=nodes, labels=nodes_labels, arrows=arrows,
                                             metrics_display=METRICS_DISPLAY),
                                   code=str_update_profile + str_update_mdr)

    profile_tap_action = CustomJS(args=dict(curr=mdr_current_data, nodes=nodes),
                                  code="""
            var indices = curr.selected.indices;
            var lost_nodes = curr.data['dr_profile_lost_id'][indices];
            
            if (lost_nodes == undefined)  // So we can use .includes
            {
                lost_nodes = []
            }
            
            for (var i=0; i< nodes.length; i++)
            {
                if (lost_nodes.includes(nodes[i].tags[0]['node_id']))
                {
                    nodes[i].line_color = 'yellow'
                }
                else 
                {
                    nodes[i].line_color = 'black'
                }
            }
                                  """)

    # set callback actions
    slider_minleaf.js_on_change('value', callback_samp_ratio)
    slider_dr.js_on_change('value', callback_dr)
    slider_maxdepth.js_on_change('value', callback_samp_ratio)
    mdr_current_data.selected.js_on_change('indices', profile_tap_action)

    bttn_save_profiles.js_on_click(CustomJS(args=dict(curr=mdr_current_data, slider_samp_ratio=slider_minleaf,
                                                      filename=filename+'_profiles'),
                                            code="""
    var csv_data = "DR,Profiles\\n";
            
            for (var i=0; i< curr.data['dr_profile_lost'].length; i++)
            {
                if (typeof curr.data['dr_profile_lost'][i] == 'string')
                {
                    csv_data += curr.data['dr_profile_x'][i] + ',' +
                                curr.data['dr_profile_lost'][i].replaceAll('<br>', ', ') + '\\n';
                }
            }         
            
            const blob = new Blob([csv_data], { type: 'text/csv;charset=utf-8;' })
            filename += '_' + slider_samp_ratio.value + '.csv';

            //addresses IE
            if (navigator.msSaveBlob) {
                navigator.msSaveBlob(blob, filename)
            } else {
                const link = document.createElement('a')
                link.href = URL.createObjectURL(blob)
                link.download = filename
                link.target = '_blank'
                link.style.visibility = 'hidden'
                link.dispatchEvent(new MouseEvent('click'))
            }
    """))


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
                plot_metrics, plot_metrics_dr,
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
            slider_maxdepth,
            slider_minleaf,
            bttn_save_profiles, sizing_mode="stretch_width"),
        [outline_boxs],
    ],
        sizing_mode="stretch_both")

    # show result
    curr_doc = curdoc()  # Document()
    curr_doc.add_root(layout_output)  # (row(inputs, plot, width=1200))

    html = file_html(layout_output, CDN, title=filename)

    path = os.path.abspath(filename + '.html')
    if not os.path.exists(path):
        os.makedirs(path)
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

    #df = pd.read_csv('simulated_data.csv')
    #x=df[['x1', 'x2']]
    #y=df['y_true'].to_numpy()
    #y_pred=df['pred_prob'].to_numpy()
    generate_mdr(x, y, y_pred)

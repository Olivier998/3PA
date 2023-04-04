from numpy import linspace, random, array, issubdtype, floating
from pandas import DataFrame
import pandas as pd
from pybase64 import b64decode
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from bokeh.io import curdoc
from bokeh.layouts import row, column, layout
from bokeh.models import Slider, Div, FileInput, MultiSelect, Select, Button, TextInput, \
    SVGIcon, Column, ColumnDataSource, Circle
from bokeh.plotting import figure
from functools import partial

import io

from utils import filter_dict
from constants import FontSize, TRUE_LABEL, PRED_LABEL, PRED_PROB
from mdr_figure_maker import generate_mdr

curr_doc = curdoc()

# Global variables
imported_data = pd.DataFrame()
selected_dependant_variables = {TRUE_LABEL: '',
                                # PRED_LABEL: '',
                                PRED_PROB: ''}
THRESHOLD = 0.5

# Tool header
tool_header = Div(text="""<b>Predictive-Performance-Precision-Analysis Tool </b>""",
                  sizing_mode="stretch_width", height=100, align="center",
                  styles={"text-align": "center", "background": "grey",
                          "font-size": FontSize.TITLE},
                  stylesheets=[":host {height: 5vw;}"])


# ## Data selection section
# Fonction to update chosen dependant variables
def update_selected_dependant_variables(variable_name):
    def update_val(attr, old, new):
        global selected_dependant_variables

        selected_dependant_variables[variable_name] = new
        predictor_variables = [var_name for var_name in imported_data.columns.tolist()
                               if var_name not in selected_dependant_variables.values()]
        predictive_variables.options = predictor_variables
        predictive_variables.value = predictor_variables

    return update_val


data_import_header = Div(text="""<b>Import Data </b>""",
                         sizing_mode="stretch_width", align="center",
                         styles={"text-align": "center", "font-size": FontSize.SUB_TITLE})

# Select true label
sel_true_label = Select(title="True label", options=imported_data.columns.tolist(), sizing_mode="stretch_width",
                        styles={"text-align": "center", "font-size": FontSize.NORMAL})
sel_true_label.on_change('value', update_selected_dependant_variables(TRUE_LABEL))

# Select predicted label
# sel_pred_label = Select(title="Predicted label", options=imported_data.columns.tolist(), sizing_mode="stretch_width",
#                         styles={"text-align": "center", "font-size": FontSize.NORMAL})
# sel_pred_label.on_change('value', update_selected_dependant_variables(PRED_LABEL))

# Select predicted probability
sel_pred_prob = Select(title="Predicted probability", options=imported_data.columns.tolist(),
                       sizing_mode="stretch_width", styles={"text-align": "center", "font-size": FontSize.NORMAL})
sel_pred_prob.on_change('value', update_selected_dependant_variables(PRED_PROB))

# Select predictive variables
predictive_variables = MultiSelect(title="Chosen predictors (second and third layer model)",
                                   sizing_mode="stretch_width",
                                   styles={"text-align": "center",
                                           "font-size": FontSize.NORMAL})  # options=imported_data.columns.tolist()


# Data Upload Button
def upload_data(attr, old, new):
    global imported_data
    decoded = b64decode(new)
    imported_data = pd.read_csv(io.StringIO(str(decoded, 'utf-8')))

    # Update selection tools
    sel_true_label.options = imported_data.columns.tolist()
    sel_true_label.value = sel_true_label.options[0]
    selected_dependant_variables[TRUE_LABEL] = sel_true_label.value
    #     sel_pred_label.options = imported_data.columns.tolist()
    sel_pred_prob.options = imported_data.columns.tolist()
    sel_pred_prob.value = sel_pred_prob.options[0]
    selected_dependant_variables[PRED_PROB] = sel_pred_prob.value
    # predictive_variables.options = imported_data.columns.tolist()


file_inputer = FileInput(title="Upload data file", accept=[".csv"],
                         styles={"text-align": "center", "font-size": FontSize.NORMAL})
file_inputer.on_change('value', upload_data)
with open("refresh.svg") as my_file:
    icon_test = my_file.read()

# Button to update Global results
bttn_update_glob_res = Button(label='Update Global Results',
                              icon=SVGIcon(svg=icon_test, size='1.5vw'),
                              align='center',
                              styles={"font-size": FontSize.BUTTON, 'width': '15vw', 'height': '3vw'},
                              stylesheets=[".bk-btn-default {font-size: 1vw; font-weight: bold;}"])
warning_bttn_update_glob_res = Div(text="""<b>WARNING</b> You must import data and choose related variables""",
                                   # sizing_mode="stretch_width",  # align="center",
                                   styles={"text-align": "center", "font-size": FontSize.NORMAL},
                                   visible=False, align='center')

# ## Global results section
# Section header
global_results_header = Div(text="""<b>Global results </b>""",  # text_font_style='bold',visage
                            sizing_mode="stretch_width",  # align="center",
                            styles={"text-align": "center", "font-size": FontSize.SUB_TITLE})


class MetricDiv:
    def __init__(self, default_string=" ", metric_fct=None, default_value="", *args, **kwargs):
        self.__div = Div(*args, **kwargs)
        self.default_string = default_string
        self.metric_fct = metric_fct
        self.value = default_value
        self.update_value(default_value)

    def update_value(self, value: str = None, **kwargs):
        if value is None:
            filtered_dict = filter_dict(self.metric_fct, **kwargs)
            value = self.metric_fct(**filtered_dict)
        self.value = round(value, 3) if issubdtype(type(value), floating) else value
        self.__div.text = self.default_string.format(self.value)

    @property
    def div(self):
        return self.__div


metrics_string = "<b><u>{metric_name}</u></b> <br> {{}}"
metrics_formatting = {"sizing_mode": "stretch_width",
                      "styles": {"text-align": "center",
                                 "font-size": FontSize.NORMAL,
                                 "padding": "1vw",
                                 "border": "1px solid black"}}

auc_value = MetricDiv(default_string=metrics_string.format(metric_name="AUC"),
                      metric_fct=roc_auc_score,
                      **metrics_formatting)

bal_acc_value = MetricDiv(default_string=metrics_string.format(metric_name="Bal Acc"),
                          metric_fct=balanced_accuracy_score,
                          **metrics_formatting)


def recall_sens_spec(pos_label):
    def _recall(**kwargs):
        return recall_score(pos_label=pos_label, **kwargs)

    return _recall


spec_value = MetricDiv(default_string=metrics_string.format(metric_name="Specificity"),
                       metric_fct=partial(recall_score, pos_label=0, zero_division=0),  # recall_sens_spec(pos_label=0),
                       **metrics_formatting)

sens_value = MetricDiv(default_string=metrics_string.format(metric_name="Sensitivity"),
                       metric_fct=partial(recall_score, pos_label=1, zero_division=0),  # recall_sens_spec(pos_label=1),
                       **metrics_formatting)


def positive_class_weight(y_true, pos_label=1, **kwargs):
    return len(y_true[y_true == pos_label]) / len(y_true)


pos_weight_value = MetricDiv(default_string=metrics_string.format(metric_name="Positive class occurence"),
                             metric_fct=positive_class_weight,
                             **metrics_formatting)

global_metrics = [auc_value, bal_acc_value, spec_value, sens_value, pos_weight_value]


# Add update function to get global values with the Update Button
def update_bttn_glob_res(event):
    if any(itm == '' for itm in selected_dependant_variables.values()) or imported_data.empty:
        warning_bttn_update_glob_res.visible = True
    else:
        # Update filename
        txt_filename.value = file_inputer.filename

        warning_bttn_update_glob_res.visible = False

        y_true = imported_data[selected_dependant_variables[TRUE_LABEL]]
        y_score = imported_data[selected_dependant_variables[PRED_PROB]]
        y_pred = [1 if y_score_i >= THRESHOLD else 0 for y_score_i in y_score]
        # imported_data[selected_dependant_variables[PRED_LABEL]]

        for metric in global_metrics:
            try:
                metric.update_value(y_true=y_true, y_pred=y_pred, y_score=y_score)
            except ValueError:
                metric.update_value('Err')
        if pos_weight_value.value != "":
            slider_weight.value = 1 - round(pos_weight_value.value, 2)


bttn_update_glob_res.on_click(update_bttn_glob_res)

# ## Generate MDR section

# fig_infos = figure(sizing_mode="scale_height", align="center")  #x_range=(0, 1), y_range=(0, 1),   styles={'height': '10', 'width': '2vw', 'align': 'center'},

# fig_infos.circle(x=0, y=0, radius=0.25)
# fig_infos.toolbar_location = None
# fig_infos.axis.visible = False
# fig_infos.grid.visible = False
# fig_infos.outline_line_color = None


generate_mdr_header = Div(text="""<b>Generate MDR </b>""",
                          sizing_mode="stretch_width", align="center",
                          styles={"text-align": "center", "font-size": FontSize.SUB_TITLE})

bttn_generate_mdr = Button(label='Generate MDR tool',
                           align='center',
                           styles={'width': '15vw', 'height': '3vw'},
                           stylesheets=[".bk-btn-default {font-size: 1vw; font-weight: bold;}"])

bttn_loading = Div(text="""<b>Loading... </b>""",
                   visible=False,
                   sizing_mode="stretch_width", align="center",
                   styles={"text-align": "center", "font-size": FontSize.NORMAL})

txt_filename = TextInput(title='Tool filename',
                         value="",
                         align='center',
                         sizing_mode="stretch_width",
                         styles={"text-align": "center", "font-size": FontSize.NORMAL, "padding": "0.5vw"},
                         stylesheets=[".bk-input {font-size: 1vw;}"])

slider_weight = Slider(start=0, end=1, value=0.5, step=0.01, title='Positive class weight (second layer model)',
                       sizing_mode="stretch_width",
                       styles={"text-align": "center", "font-size": FontSize.NORMAL, "padding": "0.5vw"},
                       stylesheets=[".bk-input {font-size: 1vw;}"])

# Outline of the 'Generate MDR' section
outline_generate_mdr = column(column(generate_mdr_header,
                                     row(column(bttn_loading, bttn_generate_mdr),
                                         txt_filename,  # Column(fig_infos, txt_filename),
                                         slider_weight,
                                         predictive_variables,
                                         align='center',
                                         sizing_mode="stretch_width"),
                                     sizing_mode="stretch_width",
                                     styles={"border": "1px solid black"}),
                              styles={"padding": "0.5vw"})


# Action for bttn_generate_mdr
def action_bttn_generate_mdr(event):
    bttn_loading.visible = True  # Doesn't work if applied after "if" verification
    if not imported_data.empty and predictive_variables.value:
        generate_mdr(x=imported_data[predictive_variables.value],
                     y=imported_data[selected_dependant_variables[TRUE_LABEL]].to_numpy(),
                     predicted_prob=imported_data[selected_dependant_variables[PRED_PROB]].to_numpy(),
                     pos_class_weight=slider_weight.value,
                     filename=txt_filename.value)
    bttn_loading.visible = False


bttn_generate_mdr.on_click(action_bttn_generate_mdr)

# Set up layout and add to document
# inputs = column(amplitude, freq)
layout_output = layout(
    [
        [tool_header],
        [column(column(data_import_header,
                       row(file_inputer, sel_true_label, sel_pred_prob,  # sel_pred_label
                           sizing_mode="stretch_width"),
                       bttn_update_glob_res,
                       warning_bttn_update_glob_res,
                       sizing_mode="stretch_width",
                       styles={"border": "1px solid black"}),
                styles={"padding": "0.5vw"})],  # , sizing_mode="stretch_width", styles={"padding": "0.5vw"})

        [column(column(global_results_header,
                       row(row(row(auc_value.div, styles={"border": "1px solid black", "padding": "0.5vw"}),
                               styles={"padding": "0.5vw"}),
                           row(row(bal_acc_value.div, styles={"border": "1px solid black", "padding": "0.5vw"}),
                               styles={"padding": "0.5vw"}),
                           # styles={"text-align": "center"}),
                           row(row(spec_value.div, styles={"border": "1px solid black", "padding": "0.5vw"}),
                               styles={"padding": "0.5vw"}),  # row(
                           row(row(sens_value.div, styles={"border": "1px solid black", "padding": "0.5vw"}),
                               styles={"padding": "0.5vw"}),
                           row(row(pos_weight_value.div, styles={"border": "1px solid black", "padding": "0.5vw"}),
                               styles={"padding": "0.5vw"}),
                           align='center'),  # styles={"text-align": "center"}),
                       sizing_mode="stretch_width", styles={"border": "1px solid black"}),
                styles={"padding": "0.5vw"})],
        outline_generate_mdr,
    ],
    sizing_mode="stretch_width"
)

curr_doc.add_root(layout_output)  # (row(inputs, plot, width=1200))
"""
# Set up data
x = linspace(0, 10, 200)
y = x**2 + random.rand(len(x))
clf = LinearRegression().fit(x.reshape(-1, 1), y)
y_pred = clf.predict(x.reshape(-1, 1))

source = ColumnDataSource(data=dict(x=x, y=y-1))
source2 = ColumnDataSource(data=dict(x=x, y=y))
#source_imported_data = ColumnDataSource(data=dict(x=DataFrame([[1], [2]], columns=['d', 'e']), y=array([0, 1])))

# Set up plot
plot = figure(x_range=(0, 10), y_range=(-2.5, 5))
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
plot.line('x', 'y', source=source2, line_width=3, line_alpha=0.6)

# Set up widgets
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1)


# Set up callbacks
def update(attrname, old, new):
    # Get the current slider values
    a = amplitude.value
    k = freq.value

    # Update the data for the new curve
    source.data = dict(x=x, y=a*x**2 + k * random.rand(len(x)))  # a*sin(k*x))
    clf = LinearRegression().fit(x.reshape(-1, 1), a*x**2 + k * random.rand(len(x)))
    source2.data = dict(x=x, y=clf.predict(x.reshape(-1, 1)))


amplitude.on_change('value', update)
freq.on_change('value', update)

# Affichage dataset
columns = [
    TableColumn(field="x", title="Employee Name"),
    TableColumn(field="y", title="Income")
]
data_table = DataTable(source=source, columns=columns, width=800)
[data_table],
        [row(column(amplitude, freq), plot, width=1200)],
"""

"""icon_test = 
<svg
   version="1.0"
   width="225.000000pt"
   height="225.000000pt"
   viewBox="0 0 225.000000 225.000000"
   preserveAspectRatio="xMidYMid meet"
   id="svg20"
   sodipodi:docname="refresh.svg"
   inkscape:version="1.2.1 (9c6d41e410, 2022-07-14)"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">
  <defs
     id="defs24" />
  <sodipodi:namedview
     id="namedview22"
     pagecolor="#ffffff"
     bordercolor="#000000"
     borderopacity="0.25"
     inkscape:showpageshadow="2"
     inkscape:pageopacity="0.0"
     inkscape:pagecheckerboard="0"
     inkscape:deskcolor="#d1d1d1"
     inkscape:document-units="pt"
     showgrid="false"
     inkscape:zoom="1.7233333"
     inkscape:cx="149.70986"
     inkscape:cy="150.29014"
     inkscape:window-width="1366"
     inkscape:window-height="697"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     inkscape:current-layer="svg20" />
  <g
     transform="translate(0.000000,225.000000) scale(0.100000,-0.100000)"
     fill="#000000"
     stroke="none"
     id="g18"
     style="fill:#008000">
    <path
       d="M934 2069 c-112 -19 -269 -87 -369 -161 -167 -124 -284 -285 -350 -482 -29 -86 -31 -121 -9 -130 9 -3 76 -6 150 -6 152 0 141 -6 184 101 67 168 238 311 426 359 146 38 295 23 433 -40 72 -34 161 -96 161 -113 0 -6 -43 -54 -95 -108 -105 -106 -115 -130 -70 -174 l24 -25 309 0 c373 0 346 -10 357 137 5 54 5 197 1 318 -8 246 -11 255 -75 255 -32 0 -46 -10 -127 -91 l-91 -91 -61 49 c-122 98 -265 166 -414 197 -84 17 -295 20 -384 5z"
       id="path14"
       style="fill:#008000" />
    <path
       d="M190 940 c-19 -19 -20 -33 -20 -338 0 -296 1 -320 18 -335 10 -10 34 -17 53 -17 29 0 46 12 126 92 l92 91 38 -34 c88 -79 212 -148 353 -195 81 -27 94 -28 270 -29 166 0 193 3 265 24 205 60 399 200 514 370 82 120 165 330 148 374 -6 15 -24 17 -145 17 -100 0 -142 -4 -150 -12 -5 -7 -22 -42 -37 -78 -138 -332 -528 -482 -860 -330 -64 29 -165 99 -165 114 0 6 43 53 95 107 105 106 115 130 70 174 l-24 25 -311 0 c-297 0 -311 -1 -330 -20z"
       id="path16"
       style="fill:#008000" />
  </g>
</svg>
"""

import pandas as pd

from datetime import date

from results_generator import produce_results

# Constants
saved_files = f'hosp/{date.today().strftime("%y%m%d")}/exp1/'

# Import data
df = pd.read_csv("../../data/sapsii/mimic_filtered_data.csv")
df = df.drop(columns=['stay_id', 'hospitalid'])

df_eicu = pd.read_csv("../../data/sapsii/eicu_filtered_data.csv")
df_eicu = df_eicu.drop(columns=['stay_id'])

# Run experiments
experiments = [
    {'exp_name': 'xgb_aucthreshold_calib_weight',
     'params': {'model': 'xgb', 'calibrate': True, 'threshold_correction': 'auc', 'class_weighting': True}},
    {'exp_name': 'xgb_nothreshold_nocalib_noweight',
     'params': {'model': 'xgb', 'calibrate': False, 'threshold_correction': None, 'class_weighting': False}},
    {'exp_name': 'xgb_nothreshold_nocalib_weight',
     'params': {'model': 'xgb', 'calibrate': False, 'threshold_correction': None, 'class_weighting': True}},
    {'exp_name': 'xgb_aucthreshold_nocalib_noweight',
     'params': {'model': 'xgb', 'calibrate': False, 'threshold_correction': 'auc', 'class_weighting': False}},
    {'exp_name': 'xgb_aucthreshold_nocalib_weight',
     'params': {'model': 'xgb', 'calibrate': False, 'threshold_correction': 'auc', 'class_weighting': True}},
    {'exp_name': 'xgb_nothreshold_calib_noweight',
     'params': {'model': 'xgb', 'calibrate': True, 'threshold_correction': None, 'class_weighting': False}},
    {'exp_name': 'xgb_nothreshold_calib_weight',
     'params': {'model': 'xgb', 'calibrate': True, 'threshold_correction': None, 'class_weighting': True}},
    {'exp_name': 'xgb_aucthreshold_calib_noweight',
     'params': {'model': 'xgb', 'calibrate': True, 'threshold_correction': 'auc', 'class_weighting': False}},

    {'exp_name': 'xgb_auprcthreshold_calib_weight',
     'params': {'model': 'xgb', 'calibrate': True, 'threshold_correction': 'auprc', 'class_weighting': True}}
]
""" Past experiments
{'exp_name': 'xgb_nothreshold_calib',
                'params': {'model': 'xgb', 'threshold_correction': None, 'calibrate': True}},
                
                
                

#                {'exp_name': 'saps_nothreshold_nocalib',
#                 'params': {'model': 'saps', 'threshold_correction': None, 'calibrate': False}},
#                {'exp_name': 'saps_nothreshold_calib',
#                 'params': {'model': 'saps', 'threshold_correction': None, 'calibrate': True}},
#                {'exp_name': 'saps_aucthreshold_nocalib',
#                 'params': {'model': 'saps', 'threshold_correction': 'auc', 'calibrate': False}},
#                {'exp_name': 'saps_aucthreshold_calib',
#                 'params': {'model': 'saps', 'threshold_correction': 'auc', 'calibrate': True}},
"""

for exp in experiments:
    path_save = saved_files + exp['exp_name'] + '/'
    produce_results(df_mimic=df, df_eicu=df_eicu, path_save=path_save, **exp['params'], do_eicu=False)

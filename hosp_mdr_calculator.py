import pandas as pd
from tqdm import tqdm

from mdr_figure_maker import generate_mdr


data_file = '../../data/saps_e.csv'
saved_files = '/hosp/hosp_'
min_hosp_samples = 200

prob_str = 'probability'
y_true_str = 'deceased'
pos_label = 1

df = pd.read_csv(data_file)

unique_hosp_id = df['hospitalid'].unique()

for hosp_id in tqdm(unique_hosp_id):
    hosp_data = df[df['hospitalid'] == hosp_id]
    if hosp_data.shape[0] >= min_hosp_samples:
        hosp_y = hosp_data[y_true_str].to_numpy()
        hosp_y_pred = hosp_data[prob_str].to_numpy()

        hosp_data = hosp_data.drop([y_true_str, prob_str], axis=1)
        hosp_filename = saved_files + str(hosp_id)

        hosp_class_imbalance = len(hosp_y[hosp_y == pos_label]) / len(hosp_y)

        generate_mdr(x=hosp_data, y=hosp_y, predicted_prob=hosp_y_pred,
                     pos_class_weight=1 - round(hosp_class_imbalance, 2), filename=hosp_filename)

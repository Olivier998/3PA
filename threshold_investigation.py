import csv
import json

from threshold_tests import test_1_maxmin, test_2_meandelta, test_3_maxslope, test_4_area

# Constants
json_file = "newtest01.json"

with open(json_file, 'r') as fp:
    data = json.load(fp)

base_metric = 'F1-Score'
dr = 'DR'

min_samp_values = data['samp_ratio']

test_values = [['samp_ratio', 'test1 maxmin', 'test2 meandelta', 'test3 maxslope', 'test4 area']]

for index, samp_ratio in enumerate(min_samp_values):
    test_values.append([samp_ratio,
                       test_1_maxmin(data['values'][index][dr], data['values'][index][base_metric]),
                       test_2_meandelta(data['values'][index][dr], data['values'][index][base_metric]),
                       test_3_maxslope(data['values'][index][dr], data['values'][index][base_metric]),
                       test_4_area(data['values'][index][dr], data['values'][index][base_metric])])

with open(json_file + '.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(test_values)

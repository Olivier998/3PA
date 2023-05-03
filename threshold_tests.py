"""

"""
import numpy as np


def test_1_maxmin(dr, values, min_dr=60):
    first_val = values[0]

    ids_greater_than_min = [x[0] for x in enumerate(dr) if x[1] >= min_dr]
    if len(ids_greater_than_min) == 0:
        return 0
    max_val = max([values[ids] for ids in ids_greater_than_min])
    return max_val - first_val


def test_2_meandelta(dr, values, min_dr=60):
    first_val = values[0]

    ids_greater_than_min = [x[0] for x in enumerate(dr) if x[1] >= min_dr]
    if len(ids_greater_than_min) == 0:
        return 0
    max_val = max([values[ids] for ids in ids_greater_than_min])
    max_index = values.index(max_val)

    if max_index == 0:
        return 0
    return (max_val - first_val) / (dr[0] - dr[max_index])


def test_3_maxslope(dr, values, min_dr=60):
    max_slope = - np.inf

    for i in range(len(values) - 1):
        if dr[i+1] >= min_dr:
            next_slope = (values[i+1] - values[i]) / (dr[i] - dr[i+1])
            if next_slope > max_slope:
                max_slope = next_slope

    return max_slope


def test_4_area(dr, values, min_dr=60):
    area = 0
    first_val = values[0]

    for i in range(len(values) - 1):
        if dr[i+1] >= min_dr:
            area += ((values[i+1] + values[i]) / 2 - first_val) * (dr[i] - dr[i+1])
        else:
            break

    return area

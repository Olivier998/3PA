"""
    @file:              data_exclusion.py
    @Author:            Olivier Lefebvre

    @Creation Date:     03/2022
    @Last modification: 03/2022

    @Description:       This file contains helpful functions to filter datasets based on the
                        Novel blending machine learning exclusion criterias
                         (https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00276-5)
"""

from pandas import DataFrame


def exclude_nbml(df: DataFrame) -> DataFrame:
    """
    Excludes rows based on the novel blending machine learning exclusion criterias
    (https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00276-5)

    :param df: Dataframe to filter
    :return: The filtered dataframe df
    """
    # Only first ICU stay
    df = df[df['unitvisitnumber'] == 1]
    df = df.drop(columns=['unitvisitnumber'])

    # Single hospitalization
    df = df[df['num_hosp'] == 1]
    df = df.drop(columns=['num_hosp'])

    # ICU stays meeting sepsis-3
    df = df[df['sepsis3'] == 1]
    df = df.drop(columns=['sepsis3'])

    # Age between 16 and 89
    df = df[(df['age'] >= 16) & (df['age'] <= 89)]

    # ICU stay duration >= 24h
    df = df[df['stay_length'] >= 1440]
    df = df.drop(columns=['stay_length'])

    # Not an organ donor
    df = df[df['organdonor'] == 0]
    df = df.drop(columns=['organdonor'])

    # Variables missing rate > 70%
    threshold = int(0.3 * df.shape[1] + 1)
    df = df.dropna(thresh=threshold)

    # Exclude non relevant variables
    df = df.drop(columns=['subject_id'])
    df = df.drop(columns=['hadm_id'])
    df = df.drop(columns=['stay_id'])

    return df

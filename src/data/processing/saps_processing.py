"""
    @file:              saps_processing.py
    @Author:            Olivier Lefebvre

    @Creation Date:     03/2022
    @Last modification: 03/2022

    @Description:       This file contains helpful functions to preprocess datasets based on the
                        SAPS-II mortality score (https://www.mdcalc.com/simplified-acute-physiology-score-saps-ii)
"""
import numpy as np
from pandas import DataFrame
from typing import Tuple


def apply_saps(df: DataFrame, convert_score: bool) -> DataFrame:
    """
    Applies saps processing to variables in the dataframe

    :param df: dataframe to process
    :param convert_score:  Boolean indicating whether keeping the SAPS-II score or original variables (between min/max)

    :return: processed dataframe
    """
    df_saps = df.copy()

    # Age
    df_saps['age'] = df_saps.apply(lambda x: transform_age(x['age'], convert_score), axis=1)

    # Bicarbonate
    df_saps['bicarbonate'] = df_saps.apply(lambda x:
                                           transform_bicarbonate(bic_min=x['bicarbonate_min'],
                                                                 convert_score=convert_score)
                                           if not np.isnan(x['bicarbonate_min']) else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['bicarbonate_min', 'bicarbonate_max'])

    # Bilirubin
    df_saps['bilirubin'] = df_saps.apply(lambda x:
                                         transform_bilirubin(bil_max=x['bilirubin_max'],
                                                             convert_score=convert_score)
                                         if not np.isnan(x['bilirubin_max']) else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['bilirubin_min', 'bilirubin_max'])

    # Bun
    df_saps['bun'] = df_saps.apply(lambda x:
                                   transform_bun(bun_max=x['bun_max'], convert_score=convert_score)
                                   if not np.isnan(x['bun_max']) else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['bun_min', 'bun_max'])

    # Chronic disease
    if convert_score:
        df_saps['chron_dis'] = df_saps.apply(lambda x:
                                             transform_chronic_disease(mets=x['mets'], hem=x['hem'], aids=x['aids'])
                                             if x[['mets', 'hem', 'aids']].notnull().all() else np.nan, axis=1)
        df_saps = df_saps.drop(columns=['mets', 'hem', 'aids'])

    # GCS
    df_saps['gcs'] = df_saps.apply(lambda x:
                                   transform_gcs(x['gcs_min'], convert_score)
                                   if not np.isnan(x['gcs_min']) else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['gcs_min'])

    # HR
    df_saps['hr'] = df_saps.apply(lambda x:
                                  transform_hr(hr_min=x['hr_min'], hr_max=x['hr_max'], convert_score=convert_score)
                                  if x[['hr_min', 'hr_max']].notnull().all() else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['hr_min', 'hr_max'])

    # Pao2fio2
    df_saps['pao2fio2'] = df_saps.apply(lambda x:
                                        transform_pao2fio2(x['pao2fio2'], convert_score)
                                        , axis=1)

    # Potassium
    df_saps['potassium'] = df_saps.apply(lambda x:
                                         transform_potassium(pot_min=x['potassium_min'], pot_max=x['potassium_max'],
                                                             convert_score=convert_score)
                                         if x[['potassium_min', 'potassium_max']].notnull().all() else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['potassium_min', 'potassium_max'])

    # Sbp
    df_saps['sbp'] = df_saps.apply(lambda x:
                                   transform_sbp(sbp_min=x['sbp_min'], sbp_max=x['sbp_max'],
                                                 convert_score=convert_score)
                                   if x[['sbp_min', 'sbp_max']].notnull().all() else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['sbp_min', 'sbp_max'])

    # Sodium
    df_saps['sodium'] = df_saps.apply(lambda x:
                                      transform_sodium(sod_min=x['sodium_min'], sod_max=x['sodium_max'],
                                                       convert_score=convert_score)
                                      if x[['sodium_min', 'sodium_max']].notnull().all() else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['sodium_min', 'sodium_max'])

    # Temperature
    df_saps['tempc'] = df_saps.apply(lambda x:
                                     transform_temperature(temp_max=x['tempc_max'], convert_score=convert_score)
                                     if not np.isnan(x['tempc_max']) else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['tempc_min', 'tempc_max'])

    # UO
    df_saps['uo'] = df_saps.apply(lambda x:
                                  transform_uo(uo=x['uo'], convert_score=convert_score)
                                  if not np.isnan(x['uo']) else np.nan, axis=1)

    # WBC
    df_saps['wbc'] = df_saps.apply(lambda x:
                                   transform_wbc(wbc_min=x['wbc_min'], wbc_max=x['wbc_max'],
                                                 convert_score=convert_score)
                                   if x[['wbc_min', 'wbc_max']].notnull().all() else np.nan, axis=1)
    df_saps = df_saps.drop(columns=['wbc_min', 'wbc_max'])

    return df_saps


def convert_saps(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Creates two dataframes based on the SAPS-II score. One dataframe contains the SAPS-II score for each variables,
    and the other one contains the variables used to obtain these scores.

    :param df: dataframe to process

    :return: A first dataframe containing the SAPS-II scores, and another one containing used variables.
    """
    df_score = apply_saps(df, True)
    df_variable = apply_saps(df, False)

    return df_score, df_variable


def transform_age(age: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's age

    :param age: age of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or variable age

    :return: related SAPS-II score
    """
    if not convert_score:
        return age

    if age < 40:
        return 0
    elif age < 60:
        return 7
    elif age < 70:
        return 12
    elif age < 75:
        return 15
    elif age < 80:
        return 16
    return 18


def transform_bicarbonate(bic_min: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's bicarbonate

    :param bic_min: Smallest bicarbonate value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or bic_min

    :return: related SAPS-II score
    """
    if not convert_score:
        return bic_min

    if bic_min < 15:
        return 6
    elif bic_min < 20:
        return 3
    return 0


def transform_bilirubin(bil_max: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's bicarbonate

    :param bic_min: Smallest bicarbonate value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or bic_min

    :return: related SAPS-II score
    """
    if not convert_score:
        return bil_max

    if bil_max >= 6:
        return 9
    elif bil_max >= 4:
        return 4
    return 0


def transform_bun(bun_max: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's bun

    :param bun_max: Greatest bun value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or original value between hr_min and hr_max

    :return: related SAPS-II score
    """
    if not convert_score:
        return bun_max

    if bun_max >= 84:
        return 10
    elif bun_max >= 28:
        return 6
    return 0


def transform_chronic_disease(mets: int, hem: int, aids: int) -> int:
    """
    Returns the SAPS-II score for a given patient's white blood cells

    :param mets: Variable indicating if patient has metastatic cancer
    :param hem: Variable indicating if patient has hematologic malignancy
    :param aids: Variable indicating if patient has AIDS

    :return: related SAPS-II score
    """
    if aids == 1:
        return 17
    elif hem == 1:
        return 10
    elif mets == 1:
        return 9
    return 0


def transform_gcs(gcs_min, convert_score: bool) -> int:
    """
    Returns the SAPS-II score for a given patient's Glasgow coma scale

    :param gcs_min: Smallest gcs value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or temp_max

    :return: related SAPS-II score
    """
    if not convert_score:
        return gcs_min

    if gcs_min < 6:
        return 26
    elif gcs_min <= 8:
        return 13
    elif gcs_min <= 10:
        return 7
    elif gcs_min <= 13:
        return 5
    return 0


def transform_hr(hr_min: int, hr_max: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's heart rate

    :param hr_min: Smallest HR value of patient
    :param hr_max: Greatest HR value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or original value between hr_min and hr_max

    :return: related SAPS-II score
    """
    if hr_min < 40:
        saps_score = 11
        saps_hr = hr_min
    elif hr_max >= 160:
        saps_score = 7
        saps_hr = hr_max
    elif hr_max >= 120:
        saps_score = 4
        saps_hr = hr_max
    elif hr_min <= 70:
        saps_score = 2
        saps_hr = hr_min
    else:
        saps_score = 0
        saps_hr = hr_max

    if convert_score:
        return saps_score
    else:
        return saps_hr


def transform_pao2fio2(pao2fio2: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's pao2/fio2 ratio

    :param pao2fio2: Smallest pao2fio2 value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or pao2fio2

    :return: related SAPS-II score
    """
    if not convert_score:
        return pao2fio2

    if pao2fio2 is None:
        return 0
    if pao2fio2 < 100:
        return 11
    elif pao2fio2 < 200:
        return 9
    return 6


def transform_potassium(pot_min: int, pot_max: int, convert_score: bool) -> int:
    """
    Returns the SAPS-II score for a given patient's systolic blood pressure

    :param pot_min: Smallest potassium value of patient
    :param pot_max: Greatest potassium value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or
            original value between pot_min and pot_max

    :return: related SAPS-II score
    """
    if pot_min < 3:
        saps_score = 3
        saps_pot = pot_min
    elif pot_max > 5:
        saps_score = 3
        saps_pot = pot_max
    else:
        saps_score = 0
        saps_pot = pot_max

    if convert_score:
        return saps_score
    return saps_pot


def transform_sbp(sbp_min: int, sbp_max: int, convert_score: bool) -> int:
    """
    Returns the SAPS-II score for a given patient's systolic blood pressure

    :param sbp_min: Smallest sbp value of patient
    :param sbp_max: Greatest sbp value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or
            original value between sbp_min and sbp_max

    :return: related SAPS-II score
    """
    if sbp_min < 70:
        saps_score = 13
        saps_sbp = sbp_min
    elif sbp_min < 100:
        saps_score = 5
        saps_sbp = sbp_min
    elif sbp_max >= 200:
        saps_score = 2
        saps_sbp = sbp_max
    else:
        saps_score = 0
        saps_sbp = sbp_max

    if convert_score:
        return saps_score
    else:
        return saps_sbp


def transform_sodium(sod_min: int, sod_max: int, convert_score: bool) -> int:
    """
    Returns the SAPS-II score for a given patient's sodium

    :param sod_min: Smallest sodium value of patient
    :param sod_max: Greatest sodium value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or
            original value between sod_min and sod_max

    :return: related SAPS-II score
    """
    if sod_min < 125:
        saps_score = 5
        saps_sod = sod_min
    elif sod_max >= 145:
        saps_score = 1
        saps_sod = sod_max
    else:
        saps_score = 0
        saps_sod = sod_min

    if convert_score:
        return saps_score
    return saps_sod


def transform_temperature(temp_max: int, convert_score: bool) -> int:
    """
    Returns the SAPS-II score for a given patient's temperature (Celcius)

    :param temp_max: Greatest temperature value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or temp_max

    :return: related SAPS-II score
    """
    if temp_max >= 39 and convert_score:
        return 3
    elif convert_score:
        return 0
    return temp_max


def transform_uo(uo: int, convert_score: bool = True) -> int:
    """
    Returns the SAPS-II score for a given patient's urine output

    :param uo: Daily urine output of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or original uo

    :return: related SAPS-II score
    """
    if not convert_score:
        return uo

    if uo < 500:
        return 11
    elif uo < 1000:
        return 4
    return 0


def transform_wbc(wbc_min: int, wbc_max: int, convert_score: bool) -> int:
    """
    Returns the SAPS-II score for a given patient's white blood cells

    :param wbc_min: Smallest wbc value of patient
    :param wbc_max: Greatest wbc value of patient
    :param convert_score: Boolean indicating whether returning SAPS-II score or
            original value between wbc_min and wbc_max

    :return: related SAPS-II score
    """
    if wbc_min < 1:
        saps_score = 12
        saps_wbc = wbc_min
    elif wbc_max >= 20:
        saps_score = 3
        saps_wbc = wbc_max
    else:
        saps_score = 0
        saps_wbc = wbc_min

    if convert_score:
        return saps_score
    return saps_wbc

"""
    @file:              data_processing.py
    @Author:            Olivier Lefebvre

    @Creation Date:     03/2022
    @Last modification: 03/2022

    @Description:       TBD
"""

import os
import pandas as pd
from settings.database import EicuSettings, MimicSettings
from settings.paths import Paths
from src.data.extraction.data_extractor import DataExtractor
from src.data.processing.data_exclusion import exclude_nbml
from src.data.processing.saps_processing import convert_saps


def process_data() -> None:
    """
    Downloads and process the mimic and eICU datasets from postgresql if not already done.
    """

    # First we get the raw datasets
    if os.path.exists(Paths.MIMIC_FILENAME_RAW):  # Mimic
        mimic_df = pd.read_csv(Paths.MIMIC_FILENAME_RAW)
    else:
        mimic_settings = MimicSettings()
        mimic_extractor = DataExtractor(*mimic_settings.get_connection_params())
        mimic_df = mimic_extractor.extract(mimic_settings.QUERY)
        mimic_df.to_csv(Paths.MIMIC_FILENAME_RAW, index=False)

    if os.path.exists(Paths.EICU_FILENAME_RAW):  # eICU
        eicu_df = pd.read_csv(Paths.EICU_FILENAME_RAW)
    else:
        eicu_settings = EicuSettings()
        eicu_extractor = DataExtractor(*eicu_settings.get_connection_params())
        eicu_df = eicu_extractor.extract(eicu_settings.QUERY)
        eicu_df.to_csv(Paths.EICU_FILENAME_RAW, index=False)

    # Data exclusion
    eicu_df = exclude_nbml(eicu_df)
    mimic_df = exclude_nbml(mimic_df)

    # Data transformation
    eicu_score, eicu_variable = convert_saps(eicu_df)
    mimic_score, mimic_variable = convert_saps(mimic_df)

    # Save processed dataframes
    eicu_score.to_csv(Paths.EICU_SAPS_SCORE, index=False)
    eicu_variable.to_csv(Paths.EICU_SAPS_VARIABLES, index=False)
    eicu_df.to_csv(Paths.EICU_ORIGINAL_VARIABLES, index=False)

    mimic_score.to_csv(Paths.MIMIC_SAPS_SCORE, index=False)
    mimic_variable.to_csv(Paths.MIMIC_SAPS_VARIABLES, index=False)
    mimic_df.to_csv(Paths.MIMIC_ORIGINAL_VARIABLES, index=False)


if __name__ == '__main__':
    process_data()

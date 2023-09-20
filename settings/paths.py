"""
    @file:              paths.py
    @Author:            Olivier Lefebvre

    @Creation Date:     03/2022
    @Last modification: 03/2022

    @Description:       Stores  every important paths within the project
"""

from os.path import dirname, join


class Paths:
    """
    Important paths used within the project
    """
    PROJECT_DIR: str = dirname(dirname(__file__))

    # Data folders
    DATA_FOLDER: str = join(PROJECT_DIR, "data")
    RAW_DATA_FOLDER: str = join(DATA_FOLDER, "raw")
    PROCESSED_DATA_FOLDER: str = join(DATA_FOLDER, "processed")
    NODE_DATA_FOLDER: str = join(DATA_FOLDER, "nodes")

    # Raw Data files
    MIMIC_FILENAME_RAW: str = join(RAW_DATA_FOLDER, "raw_mimic.csv")
    EICU_FILENAME_RAW: str = join(RAW_DATA_FOLDER, 'raw_eicu.csv')

    # Processed Data files
    MIMIC_SAPS_SCORE: str = join(PROCESSED_DATA_FOLDER, 'sapsii_score_mimic.csv')
    MIMIC_SAPS_VARIABLES: str = join(PROCESSED_DATA_FOLDER, 'sapsii_variable_mimic.csv')
    MIMIC_ORIGINAL_VARIABLES: str = join(PROCESSED_DATA_FOLDER, 'original_mimic.csv')

    EICU_SAPS_SCORE: str = join(PROCESSED_DATA_FOLDER, 'sapsii_score_eicu.csv')
    EICU_SAPS_VARIABLES: str = join(PROCESSED_DATA_FOLDER, 'sapsii_variable_eicu.csv')
    EICU_ORIGINAL_VARIABLES: str = join(PROCESSED_DATA_FOLDER, 'original_eicu.csv')

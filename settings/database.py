"""
    @file:              database.py
    @Author:            Olivier Lefebvre

    @Creation Date:     03/2022
    @Last modification: 03/2022

    @Description:       Stores databases connection parameters for the mimic-IV and the eICU databases
"""

from typing import Tuple


class EicuSettings:
    """
    Stores the information allowing connection to the mimic database
    """
    USER: str = 'lefo2801'
    PASSWORD: str = 'medomics101'
    DATABASE: str = 'eicu'  # Ex. 'my_database'
    HOST: str = 'localhost'  # Ex. 'localhost'
    PORT: str = '5437'  # Ex. '5437'
    SCHEMA: str = 'eicu_crd'  # Ex. 'public'
    QUERY: str = """select patientunitstayid as stay_id
    , hadm_id
    , subject_id
    , hospitalid
    , deceased --case when deceased_hosp is null then deceased_unit else deceased_hosp end as deceased
    , age
    , stay_length
    , organdonor
    , bicarbonate_min
    , bicarbonate_max
    , bilirubin_min
    , bilirubin_max
    , potassium_min
    , potassium_max
    , sodium_min
    , sodium_max
    , bun_min
    , bun_max
    , wbc_min
    , wbc_max
    , pao2fio2
    , gcs_min
    , hr_min
    , hr_max
    , tempc_min
    , tempc_max
    , sbp_min
    , sbp_max
    , uo
    , aids
    , hem
    , mets
    , admissiontype
    , sepsis3
    , unitvisitnumberrevised as unitvisitnumber
    , num_hosp
    from eicu_crd.sapsii_data_24"""  # Ex. 'select * from {SCHEMA}.{TABLE}'

    def get_connection_params(self) -> Tuple[str, str, str, str, str, str]:
        """
        Returns parameters to allow a database connection
        :return:
        """
        return self.USER, self.PASSWORD, self.DATABASE, self.HOST, self.PORT, self.SCHEMA


class MimicSettings:
    """
    Stores the information allowing connection to the mimic database
    """
    USER: str = 'lefo2801'
    PASSWORD: str = 'medomics101'
    DATABASE: str = 'mimic_iv'  # Ex. 'my_database'
    HOST: str = 'localhost'  # Ex. 'localhost'
    PORT: str = '5437'  # Ex. '5437'
    SCHEMA: str = 'mimic_derived'  # Ex. 'public'
    QUERY: str = """select stay_id
    , hadm_id
    , subject_id
    , hospitalid
    , deceased
    , age
    , stay_length
    , organdonor
    , bicarbonate_min
    , bicarbonate_max
    , bilirubin_min
    , bilirubin_max
    , potassium_min
    , potassium_max
    , sodium_min
    , sodium_max
    , bun_min
    , bun_max
    , wbc_min
    , wbc_max
    , pao2fio2
    , gcs_min
    , hr_min
    , hr_max
    , tempc_min
    , tempc_max
    , sbp_min
    , sbp_max
    , uo
    , aids
    , hem
    , mets
    , admissiontype
    , sepsis3
    , unitvisitnumber
    , num_hosp
    from mimic_derived.sapsii_data_24 """  # Ex. 'select * from {SCHEMA}.{TABLE}'

    def get_connection_params(self) -> Tuple[str, str, str, str, str, str]:
        """
        Returns parameters to allow a database connection
        :return:
        """
        return self.USER, self.PASSWORD, self.DATABASE, self.HOST, self.PORT, self.SCHEMA

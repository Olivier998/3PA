"""
    @file:              data_extractor.py
    @Author:            Olivier Lefebvre

    @Creation Date:     01/2022
    @Last modification: 03/2022

    @Description:       TBD
"""

import pandas as pd
from sqlalchemy import create_engine
from typing import Any, Dict, List, Optional, Tuple


class DataExtractor:
    """
    TBD
    """

    def __init__(self,
                 user: str,
                 password: str,
                 database: str,
                 host: str = 'localhost',
                 port: str = '5432',
                 schema: str = 'public'):
        """
        TBD

        :param user:
        :param password:
        :param database:
        :param host:
        :param port:
        :param schema:
        """
        # self.__conn, self.__cur = DataExtractor._connect(user, password, database, host, port)
        # Create the connection
        self._connect(user, password, database, host, port)

        self.__schema = schema

    def extract(self, query: str):
        """
        A faire
        returns pandas dataframe of data
        """

        try:
            df = pd.read_sql_query(query, con=self.cnx)

        except Exception as e:
            print(e)
            raise
        return df

    def _connect(self,
                 user: str,
                 password: str,
                 database: str,
                 host: str,
                 port: str) -> Tuple[Any, Any]:
        """
        Creates a connection with a database

        Args:
            user: username to access the database
            password: password linked to the user
            database: name of the database
            host: database host address
            port: connection port number

        Returns: connection, cursor
        """

        postgres_str = (f"postgresql://{user}:{password}@{host}:{port}/{database}")

        try:
            self.cnx = create_engine(postgres_str)

        except Exception as e:
            print(e)
            raise

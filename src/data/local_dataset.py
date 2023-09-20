"""
    @file:              local_dataset.py
    @Author:            Olivier Lefebvre

    @Creation Date:     03/2022
    @Last modification: 03/2022

    @Description:       Defines the class related to datasets.
"""


import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Union


class LocalDataset(Dataset):
    """
    Custom dataset class.
    """
    def __init__(self, paths: List[str] = None,
                 *,
                 df: pd.DataFrame = None,
                 x: Tuple[pd.DataFrame, pd.Series] = None,
                 y: Tuple[pd.Series, int] = None) -> None:
        """
        Sets attributes of our custom dataset class.

        :param paths: Paths of the csv files.
        """
        assert paths is not None or df is not None, "Cannot initialize a LocalDataset without specifying data."
        assert paths is None or df is None, "Must initialize a LocalDataset with a list of strings of with a df."

        if paths is not None:
            dataframes = [pd.read_csv(file) for file in paths]
            self._df = pd.concat(dataframes)
            self._df = self._df.drop(columns=['hospitalid'])
            self.x = self._df.drop(columns=['deceased'])
            self.y = self._df['deceased']

        else:
            self._df = df
            self.x = x
            self.y = y

    def __len__(self) -> int:
        return self._df.shape[0]

    def __getitem__(self, idx: Union[int, List[int]]
                    ) -> 'LocalDataset':  # Union[Tuple[pd.Series, int], Tuple[pd.DataFrame, pd.Series]]
        df = self._df.iloc[idx].copy()
        x = self.x.iloc[idx]
        y = self.y.iloc[idx]

        return LocalDataset(df=df, x=x, y=y)

    def fill_missing(self, mean: float = None) -> None:
        """
        Fills missing values in the dataset

        :param mean:
        """
        if mean is not None:
            self.x = self.x.fillna(mean)
        else:
            self.x = self.x.fillna(self.x.mean())

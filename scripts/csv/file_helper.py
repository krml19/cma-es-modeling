import pandas as pd
from os import makedirs
# import pandas_profiling
import os
from enum import Enum


class Paths(Enum):
    train = "data/train/"
    test = "data/test/"
    valid = "data/valid/"

    def path(self, filename: str):
        return self.value + filename


def __write_to_file(filename: str, df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame)
    df.to_csv(filename, index=False, float_format='%.10f')


def write_data_frame(df: pd.DataFrame, path: str, filename: str, extension: str='.csv'):
    path = os.path.abspath(path)
    makedirs(path, exist_ok=True)
    filename = path + "/" + filename + extension
    __write_to_file(filename=filename, df=df)

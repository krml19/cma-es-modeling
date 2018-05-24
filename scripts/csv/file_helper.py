import pandas as pd
from os import makedirs
# import pandas_profiling
import os
from enum import Enum


class Paths(Enum):
    train = "datasets/training"
    test = "datasets/test"
    valid = "datasets/validation"
    tables = "latex/tables/"

    def path(self, filename: str):
        return self.value + filename


def concat_filename(path: str, filename: str, extension: str) -> str:
    path = os.path.abspath(path)
    makedirs(path, exist_ok=True)
    filename = path + "/" + filename + extension
    return filename

def __write_to_file(filename: str, df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame)
    df.to_csv(filename, index=False, float_format='%.10f')


def write_data_frame(df: pd.DataFrame, path: str, filename: str, extension: str='.csv'):
    filename = concat_filename(path=path, filename=filename, extension=extension)
    __write_to_file(filename=filename, df=df)


def write_tex_table(filename: str, data: str=None, extension: str='.tex', path: str=Paths.tables.value):
    filename = path + filename + extension
    with open(filename, "w") as file:
        file.write(data)

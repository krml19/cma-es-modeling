import os
import pandas as pd

from enum import Enum


class Paths(Enum):
    train = "datasets/training"
    test = "datasets/test"
    valid = "datasets/validation"
    tables = "latex/tables/"
    best_models = "results/"

    def path(self, filename: str):
        return self.value + filename


def concat_filename(path: str, filename: str, extension: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    filename = path + "/" + filename + extension
    return filename

def __write_to_file(filename: str, df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame)
    df.to_csv(filename, index=False, float_format='%.10f', compression='xz')


def write_data_frame(df: pd.DataFrame, path: str, filename: str, extension: str='.csv.xz'):
    filename = concat_filename(path=path, filename=filename, extension=extension)
    __write_to_file(filename=filename, df=df)


def write_file(filename: str, data: str=None, extension: str= '.tex', path: str=Paths.tables.value):
    filename = path + filename + extension
    with open(filename, "w", encoding='utf-8') as file:
        file.write(data)


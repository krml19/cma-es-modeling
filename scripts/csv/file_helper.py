import pandas as pd
from os import listdir, makedirs, path
import pandas_profiling
import os


def write_to_csv(path: str, df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame)
    df.to_csv("{}.csv".format(path), index=False, float_format='%.10f')


def write_description(filename_with_dir: str, df: pd.DataFrame):
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile="{}.html".format(filename_with_dir))


def write_train_file(df: pd.DataFrame, path: str='data/train/', filename='train', extension='.csv'):
    path = os.path.abspath(path)
    makedirs(path, exist_ok=True)

    file_numbers = [int(f.replace('_','.').split('.')[1]) for f in listdir(path)
             if len(f.replace('_','.').split('.')) == 3 and f.replace('_','.').split('.')[0] == filename]
    file_number = max(file_numbers) + 1 if len(file_numbers) > 0 else 0
    file_name = "{}/{}_{}".format(path, filename, file_number)

    write_to_csv(path="{}".format(file_name), df=df)
    write_description(file_name, df=df)

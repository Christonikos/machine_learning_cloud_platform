import pandas as pd
from path import data_folder_path


def df_cleaning(df):
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    df.drop_duplicates(ignore_index=True, inplace=True)
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)

    return df


def clean():
    df = pd.read_csv(data_folder_path("raw", "census.csv"), skipinitialspace=True)  # noqa: E501
    df = df_cleaning(df)
    df.to_csv(data_folder_path("preprocessed", "preprocessed.csv"))

"""
Test module for data_cleaning module
"""
import pandas as pd
from path import data_folder_path
import pytest


@pytest.fixture
def load_data():
    """
    Load preprocessed census dataset as pandas dataframe

    """
    df = pd.read_csv(data_folder_path("preprocessed", "preprocessed.csv"))
    return df


def test_for_nulls(load_data):
    """
    Test to make sure there are no missing values

    Args:
        load_data ([type]): [description]
    """
    assert load_data.shape == load_data.dropna().shape


def test_for_nulls_II(load_data):
    """
    Data is assumed to have no question marks value
    """
    assert '?' not in load_data.values

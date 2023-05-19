"""
Test module for model_dev
"""
# Importing the necessary modules
import math
import os.path
from sklearn.model_selection import train_test_split
from joblib import load
import pandas as pd
import pytest
import numpy as np
from path import data_folder_path
from model_dev import inference, compute_model_metrics
from data import process_data


df = pd.read_csv(data_folder_path("preprocessed", "preprocessed.csv"))


def load_prediction_and_y():
    """
    Load prediction and y_test to use in below tests

    """
    _, test = train_test_split(df, test_size=0.20, random_state=20)  # noqa: E501

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    trained_model = load(data_folder_path("model", "model.joblib"))
    encoder = load(data_folder_path("encoder", "encoder.joblib"))
    lb = load(data_folder_path("lb", "lb.joblib"))

    X_test, y_test, _, _ = process_data(test,
                                        categorical_features=cat_features,
                                        label="salary",
                                        encoder=encoder,
                                        lb=lb,
                                        training=False)

    prediction = inference(trained_model, X_test)

    return prediction, y_test


def test_check_inference_type_length_and_values():
    """
    Function that checks that the type and size of the prediciton is correct.
    Also checks that both classes are present in the prediction.
    Args:
        load_data ([pandas.core.frame.DataFrame]): [census preprocessed dataset loaded in pandas]  # noqa: E501
    """
    prediction, _ = load_prediction_and_y()
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == math.ceil(len(df) * 0.2)
    assert list(np.unique(prediction)) == [0, 1]


def test_fbeta_precision_recall():
    """
    Checks that there is not a drift in the scoring metrics
    Args:
        load_data ([pandas.core.frame.DataFrame]): [census preprocessed dataset loaded in pandas]  # noqa: E501
    """
    prediction, y_test = load_prediction_and_y()

    precision, recall, fbeta = compute_model_metrics(y_test, prediction)
    assert precision > 0.72 and recall > 0.55 and fbeta > 0.62


def test_slices():
    """
    Checks that slices functions outputs the scoring slices in the .txt file
    """
    assert os.path.isfile(data_folder_path("slices", "slice_output.txt"))

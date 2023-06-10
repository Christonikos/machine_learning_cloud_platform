#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

import os
import requests
import tempfile
import shutil
import pandas as pd
import pytest
import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score
from src.data_injestion_01 import load_raw_data
from src.model_training_02 import (
    compute_model_metrics,
    data_loader,
    train_and_save_model,
)


# =============================================================================
# DATA INJESTION TESTS (01)
# =============================================================================
@pytest.fixture
def create_temp_csv_files():
    """
    This fixture creates a temporary directory with a 'raw' subdirectory
    and a sample CSV file inside it. The temporary directory is cleaned up
    after the test is executed.

    :return: The path to the temporary directory containing the sample CSV file.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create a 'raw' subdirectory inside the temporary directory
    os.makedirs(os.path.join(temp_dir, "raw"))

    # Define the data for the sample CSV file
    test_data = {"A": [1, 2, 3], "B": [4, 5, 6]}

    # Create a pandas DataFrame from the test data
    df = pd.DataFrame(test_data)

    # Save the DataFrame as a CSV file inside the 'raw' subdirectory
    df.to_csv(os.path.join(temp_dir, "raw", "test.csv"), index=False)

    # Yield the path to the temporary directory so that it can be used by the test functions
    yield temp_dir

    # Clean up the temporary directory and its contents after the test is executed
    shutil.rmtree(temp_dir)


def test_load_raw_data(create_temp_csv_files):
    """
    Test the load_raw_data function with a valid path containing a CSV file.
    Assert that the function returns the correct DataFrame.
    """
    path_to_data = create_temp_csv_files
    data_frame = load_raw_data(path_to_data)

    expected_data = {"A": [1, 2, 3], "B": [4, 5, 6]}

    expected_data_frame = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(data_frame, expected_data_frame)


def test_load_raw_data_nonexistent_directory():
    """
    Test the load_raw_data function with a nonexistent directory path.
    Assert that the function returns None.
    """
    path_to_data = "/nonexistent/directory"
    data_frame = load_raw_data(path_to_data)
    assert data_frame is None


def test_load_raw_data_no_csv_files(create_temp_csv_files):
    """
    Test the load_raw_data function with a valid path but without any CSV files.
    Assert that the function returns None.
    """
    # Remove the CSV file created by the fixture
    os.remove(os.path.join(create_temp_csv_files, "raw", "test.csv"))

    path_to_data = create_temp_csv_files
    data_frame = load_raw_data(path_to_data)
    assert data_frame is None


# =============================================================================
# MODEL TRAINING TESTS (02)
# =============================================================================


def test_compute_model_metrics():
    """
    Test the compute_model_metrics function.
    This function checks if the returned precision, recall and fbeta scores match
    the expected scores computed using sklearn's functions.
    """
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 1, 0]
    beta = 1
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred, beta=beta)

    assert precision == precision_score(
        y_true, y_pred
    ), "Mismatch in precision scores"
    assert recall == recall_score(y_true, y_pred), "Mismatch in recall scores"
    assert fbeta == fbeta_score(
        y_true, y_pred, beta=beta
    ), "Mismatch in fbeta scores"


def test_data_loader():
    """
    Test the data_loader function.
    This function checks if the returned object is a pandas DataFrame.
    """
    data = data_loader(data_type="preprocessed", data_path="data")
    assert isinstance(
        data, pd.DataFrame
    ), "Loaded data is not a pandas DataFrame"


def test_load_saved_model():
    """
    Test the load_saved_model function.
    This function checks if a trained model can be successfully loaded.
    """
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "trained_model.joblib",
        )
    )

    try:
        model = joblib.load(model_path)
        assert model is not None, "Failed to load the saved model"
    except Exception as e:
        pytest.fail(f"load_saved_model raised exception: {e}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

import os
import tempfile
import shutil
import pandas as pd

import pytest

from data_injestion_01 import load_raw_data

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
    os.makedirs(os.path.join(temp_dir, 'raw'))

    # Define the data for the sample CSV file
    test_data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }

    # Create a pandas DataFrame from the test data
    df = pd.DataFrame(test_data)

    # Save the DataFrame as a CSV file inside the 'raw' subdirectory
    df.to_csv(os.path.join(temp_dir, 'raw', 'test.csv'), index=False)

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

    expected_data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }

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
    os.remove(os.path.join(create_temp_csv_files, 'raw', 'test.csv'))

    path_to_data = create_temp_csv_files
    data_frame = load_raw_data(path_to_data)
    assert data_frame is None

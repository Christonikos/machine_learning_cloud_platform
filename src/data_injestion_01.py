#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that loads the raw data, preprocess it and stores it into the
preprocessed folder for model training.

@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import argparse
import logging
from typing import Optional

import pandas as pd


# =============================================================================
# UTILITY FUNCTIONS/CLASSES
# =============================================================================
logger = logging.getLogger(__name__)


def load_raw_data(path_to_data: str) -> Optional[pd.DataFrame]:
    """
    Load the first CSV file from the 'raw' subdirectory of the given path.

    :param path_to_data: The path to the directory containing the 'raw' subdirectory.
    :return: A pandas DataFrame containing the contents of the first CSV file,
    or None if no CSV files are found.
    """
    path_to_raw = os.path.join(path_to_data, 'raw')

    if not os.path.exists(path_to_raw):
        logger.error(f"Directory not found: {path_to_raw}")
        return None

    csv_files = [
        file for file in os.listdir(path_to_raw)
        if not file.startswith('.') and file.endswith('.csv')
    ]

    if not csv_files:
        logger.warning(f"No CSV files found in {path_to_raw}")
        return None

    file_name = os.path.join(path_to_raw, csv_files[0])
    logger.info(f"Loading CSV file: {file_name}")

    try:
        data_frame = pd.read_csv(file_name)
        # Remove white space from column names
        data_frame.columns = [col.replace(' ', '') for col in data_frame.columns]
    except Exception as e:
        logger.error(f"Error reading CSV file {file_name}: {e}")
        return None

    return data_frame


# =============================================================================
# MAIN CALLER
# =============================================================================
path2data = os.path.join('..','data')
# load the raw data
raw_data = load_raw_data(path2data)


def main(args):
    pass


# =============================================================================
# WRAPER
# =============================================================================
if __name__ == '__main__':

    # =========================================================================
    # ARGPARSE INPUTS
    # =========================================================================
    parser = argparse.ArgumentParser(
        description=" ")
    parser.add_argument('-arg1',
                        '--arg1_long',
                        default='',
                        help=" ")
    args = parser.parse_args()
    pass

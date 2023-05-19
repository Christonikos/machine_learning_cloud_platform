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
import logging
from typing import Optional
from ydata_profiling import ProfileReport
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from datetime import datetime
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
    path_to_raw = os.path.join(path_to_data, "raw")

    if not os.path.exists(path_to_raw):
        logger.error(f"Directory not found: {path_to_raw}")
        return None

    csv_files = [
        file
        for file in os.listdir(path_to_raw)
        if not file.startswith(".") and file.endswith(".csv")
    ]

    if not csv_files:
        logger.warning(f"No CSV files found in {path_to_raw}")
        return None

    file_name = os.path.join(path_to_raw, csv_files[0])
    logger.info(f"Loading CSV file: {file_name}")

    try:
        data_frame = pd.read_csv(file_name)
        # Remove white space from column names
        data_frame.columns = [col.replace(" ", "") for col in data_frame.columns]
    except Exception as e:
        logger.error(f"Error reading CSV file {file_name}: {e}")
        return None

    return data_frame


def binarize_label(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Binarize the label."""
    raw_data["salary"] = raw_data["salary"].str.strip()
    raw_data["salary"] = raw_data["salary"].replace({"<=50K": -1, ">50K": 1})
    return raw_data


def generate_profile_report(
    raw_data: pd.DataFrame, report_dir: str = "../reports"
) -> None:
    """Generate a pandas profiling report and save it as an HTML file."""
    profile = ProfileReport(raw_data, title="Pandas Profiling Report", explorative=True)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    profile.to_file(os.path.join(report_dir, "eda_report.html"))


def data_inspection(raw_data: pd.DataFrame, report_dir: str = "../reports") -> None:
    """Script for data inspection. Currently generates a pandas Profile report"""
    generate_profile_report(raw_data, report_dir)


def save_preprocessing_info(
    raw_data: pd.DataFrame,
    preprocessed_data: pd.DataFrame,
    categorical_features,
    continuous_columns,
    original_data_path: str,
    preprocessed_data_path: str,
) -> None:
    original_shape = raw_data.shape
    preprocessed_shape = preprocessed_data.shape
    num_cat_columns = len(categorical_features)
    num_cont_columns = len(continuous_columns)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = f"""
        Data Preprocessing Information:

        1. Date and Time: {current_datetime}


        2. Original Data:
            - Shape: {original_shape}
            - Loaded from: {original_data_path}

        3. Preprocessed Data:
            - Shape: {preprocessed_shape}
            - Stored at: {preprocessed_data_path}

        4. Features:
            - Number of categorical columns: {num_cat_columns}
            - Categorical columns: {categorical_features}
            - Number of continuous columns: {num_cont_columns}
            - Continuous columns: {continuous_columns}

        5. Preprocessing Steps:
            - Remove white spaces from the column names
            - Binarize the target column (salary)
            - Drop the columns: education-num, capital-gain, capital-loss
            - Apply one-hot encoding to the categorical features
            - Scale the continuous features using RobustScaler
            - Concatenate the preprocessed categorical and continuous features
            - Add the target column (salary) to the preprocessed data
            """

    report_dir = "../reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    file_path = os.path.join(report_dir, "data_injestion_report.txt")
    with open(file_path, "w") as file:
        file.write(info)

    print(f"Preprocessing information saved to {file_path}")


def data_preprocessing(raw_data: pd.DataFrame) -> None:
    """
    Notes from the Pandas Profiling:

    1. education-num is highly overall correlated with education	High correlation
    2. education is highly overall correlated with education-num	High correlation
    3. relationship is highly overall correlated with sex	High correlation
    4. sex is highly overall correlated with relationship	High correlation
    5. race is highly imbalanced (65.6%)	Imbalance
    6. native-country is highly imbalanced (82.5%)	Imbalance
    7. capital-gain has 29849 (91.7%) zeros	Zeros
    8. capital-loss has 31042 (95.3%) zeros	Zeros
    """

    # Prepare categorical and continuous pipelines
    categorical_pipeline = Pipeline(steps=[("onehot", OneHotEncoder())])
    continuous_pipeline = Pipeline(steps=[("scaler", RobustScaler())])

    # Remove white space from the column naming
    raw_data.columns = raw_data.columns.str.strip()
    raw_data = binarize_label(raw_data.copy())

    # Points 1 & 2, 7 & 8
    raw_data.drop(
        columns=["education-num", "capital-gain", "capital-loss"], inplace=True
    )

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    target = raw_data.pop("salary")
    X_continuous = raw_data.drop(columns=categorical_features)

    # Create a column transformer to handle both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("cont", continuous_pipeline, X_continuous.columns),
        ]
    )
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(raw_data)

    # Get the transformed categorical features and their column names
    X_categorical = preprocessor.named_transformers_["cat"]["onehot"].transform(
        raw_data[categorical_features]
    )
    cat_columns = list(
        preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
            categorical_features
        )
    )

    cont_columns = list(X_continuous.columns)

    # Get the transformed continuous features and their column names
    X_continuous = preprocessor.named_transformers_["cont"]["scaler"].transform(
        X_continuous
    )

    # Create DataFrames for the transformed features and reset their indices
    cat_df = pd.DataFrame(X_categorical.toarray(), columns=cat_columns).reset_index(
        drop=True
    )  # Convert sparse matrix to array
    cont_df = pd.DataFrame(X_continuous, columns=cont_columns).reset_index(drop=True)

    # Reset the index for the target column
    target = target.reset_index(drop=True)

    # Combine the transformed features and the target column into a single DataFrame
    preprocessed_data = pd.concat([cat_df, cont_df], axis=1)

    # TODO: Make sure there are no '?' columns left in the data and also that
    # the 'unamed:0' column is dropped
    preprocessed_data["salary"] = target

    # save the preprocessed dataframe
    preprocessed_data_path = "../data/preprocessed"
    if not os.path.exists(preprocessed_data_path):
        os.makedirs(preprocessed_data_path)
    fname = os.path.join(preprocessed_data_path, "census_preprocessed.csv")
    preprocessed_data.to_csv(fname)
    # Reporting
    # # Save the preprocessing information to a .txt file
    save_preprocessing_info(
        raw_data,
        preprocessed_data,
        categorical_features,
        cont_columns,
        "path/to/original_data.csv",
        preprocessed_data_path,
    )


# =============================================================================
# MAIN CALLER
# =============================================================================
if __name__ == "__main__":
    path2data = os.path.join("..", "data")
    # load the raw data
    raw_data = load_raw_data(path2data)
    # perform data inspection and generate an html report
    data_inspection(raw_data)
    # preprocsess and sava the dataframe
    data_preprocessing(raw_data)

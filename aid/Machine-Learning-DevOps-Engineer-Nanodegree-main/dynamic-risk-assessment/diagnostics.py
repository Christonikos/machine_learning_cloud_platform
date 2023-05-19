
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from joblib import load
import subprocess
import sys


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['prod_deployment_path'])
test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

# Function to get model predictions


def model_predictions(test_data):
    # read the deployed model and a test dataset, calculate predictions

    trained_model = load(os.path.join(output_model_path, "trainedmodel.pkl"))

    X_test = test_data.loc[:, ["lastmonth_activity", "lastyear_activity",
                           "number_of_employees"]].values.reshape(-1, 3)
    y_test = test_data["exited"].values.reshape(-1, 1).ravel()

    y_pred = trained_model.predict(X_test)

    # return value should be a list containing all predictions
    return y_pred

# Function to get summary statistics


def dataframe_summary():
    # calculate summary statistics here
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
    ]

    summary_statistics = []
    for column in numeric_columns:
        summary_statistics.append([column, "mean", df[column].mean()])
        summary_statistics.append([column, "median", df[column].median()])
        summary_statistics.append(
            [column, "standard deviation", df[column].std()])

    # return value should be a list containing all summary statistics
    return summary_statistics

# Function to count missing values


def missing_values():
    # calculate percentage of the missing values by columns
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    pct_missing = list(df.isna().sum(axis=1)/df.shape[0])

    return pct_missing

# Function to get timings


def execution_time():
    # calculate timing of training.py and ingestion.py
    all_timings = []
    for procedure in ["training.py", "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing = timeit.default_timer() - starttime
        all_timings.append([procedure, timing])
    return str(all_timings)  # return a list of 2 timing values in seconds

# Function to check dependencies


def outdated_packages_list():
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions(test_data)
    dataframe_summary()
    missing_values()
    execution_time()
    outdated_packages_list()

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from joblib import load
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

# Function for model scoring

test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

ML_model = load(os.path.join(output_model_path, "trainedmodel.pkl"))


def score_model(test_data, ML_model):

    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    X_test = test_data.loc[:, ["lastmonth_activity", "lastyear_activity",
                               "number_of_employees"]].values.reshape(-1, 3)
    y_test = test_data["exited"].values.reshape(-1, 1).ravel()

    y_pred = ML_model.predict(X_test)

    f1 = metrics.f1_score(y_test, y_pred)

    with open(os.path.join(output_model_path, "latestscore.txt"), "w") as score:
        score.write(str(f1) + "\n")
    return f1


if __name__ == "__main__":
    score_model(test_data, ML_model)

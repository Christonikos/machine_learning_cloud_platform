from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from joblib import dump
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.preprocessing import OneHotEncoder


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

# Function for training the model


def train_model():

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='ovr', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    os.chdir(dataset_csv_path)
    df = pd.read_csv('finaldata.csv')
    X = df.loc[:, ["lastmonth_activity", "lastyear_activity",
                   "number_of_employees"]].values.reshape(-1, 3)
    y = df["exited"].values.reshape(-1, 1).ravel()
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    os.chdir(os.path.dirname(os.getcwd()))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    os.chdir(model_path)
    with open("trainedmodel.pkl", "wb") as trained_model:
        dump(model, trained_model)


if __name__ == "__main__":
    train_model()

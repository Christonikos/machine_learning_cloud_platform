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
import shutil


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)
output_model_path = os.path.join(config['output_model_path'])
output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

model = str(os.path.join(os.getcwd(), output_model_path, "trainedmodel.pkl"))
# function for deployment


def store_model_into_pickle(model):
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    score = str(os.path.join(
        os.getcwd(), output_model_path, "latestscore.txt"))
    ing_data = str(os.path.join(
        os.getcwd(), output_folder_path, "ingestedfiles.txt"))

    shutil.move(score, str(
        (os.path.join(prod_deployment_path, "latestscore.txt"))))
    shutil.move(ing_data, str((os.path.join(
        prod_deployment_path, "ingestedfiles.txt"))))
    shutil.move(model, str(
        (os.path.join(prod_deployment_path, "trainedmodel.pkl"))))


if __name__ == '__main__':
    store_model_into_pickle(model)

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
from joblib import load
# import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_values, missing_values, execution_time, outdated_packages_list
from scoring import score_model


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
# output_model_path = os.path.join(config['prod_deployment_path'])
test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
ML_model = load(os.path.join(output_model_path, "trainedmodel.pkl"))

# Prediction Endpoint


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    pred = model_predictions(test_data)
    return str(pred)  # add return value for prediction outputs

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # check the score of the deployed model
    score = score_model(test_data, ML_model)
    return str(score)  # add return value (a single F1 score number)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    summary = dataframe_summary()
    return str(summary)  # return a list of all calculated summary statistics

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    timing = execution_time()
    missing_data = missing_values()
    outdated_list = outdated_packages_list()

    # add return value for all diagnostics
    return str(f'Execution time: {timing} \n Missing data(%): {missing_data} \n Outdated packages: {outdated_list}')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)

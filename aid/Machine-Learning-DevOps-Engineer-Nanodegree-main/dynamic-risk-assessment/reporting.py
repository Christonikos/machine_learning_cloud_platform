import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import json
import os
from joblib import load

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_folder_path = os.path.join(config['output_model_path'])
output_model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))


# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    trained_model = load(os.path.join(output_model_path, "trainedmodel.pkl"))

    X_test = test_data.loc[:, ["lastmonth_activity", "lastyear_activity",
                           "number_of_employees"]].values.reshape(-1, 3)
    y_test = test_data["exited"].values.reshape(-1, 1).ravel()

    trained_model = load(os.path.join(output_model_path, "trainedmodel.pkl"))

#     y_pred = model_predictions(test_data)
#     conf_mtrx = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(trained_model, X_test, y_test)

    # write the confusion matrix to the workspace
    plt.savefig(os.path.join(model_folder_path, "confusionmatrix.png"))


if __name__ == '__main__':
    score_model()

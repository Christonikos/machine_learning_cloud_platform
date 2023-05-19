#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

import os
import pandas as pd
import numpy as np
from joblib import dump
import json
from datetime import datetime
import logging
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    confusion_matrix,
)


logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS/CLASSES
# =============================================================================
def data_loader(
    data_type: str = "preprocessed", data_path: str = "../data"
) -> pd.DataFrame():
    """
    Function to load a .csv file from a specified directory. It will load the first .csv file
    it finds in the specified directory. It also removes any whitespace from column names.

    Parameters:
    data_type (str): Type of data to be loaded. This is used to define the sub-directory
                     from which the data will be loaded. Default is 'preprocessed'.
    data_path (str): The directory path where the data files are located.
                     Default is '../data'.

    Returns:
    pd.DataFrame: The loaded dataframe. Returns None if no .csv file is found or an error
                  occurred while loading.
    """
    path2data = os.path.join(data_path, data_type)
    if not os.path.exists(path2data):
        logger.error(f"Directory not found: {path2data}")
        return None

    csv_files = [
        file
        for file in os.listdir(path2data)
        if not file.startswith(".") and file.endswith(".csv")
    ]

    if not csv_files:
        logger.warning(f"No CSV files found in {path2data}")
        return None

    file_name = os.path.join(path2data, csv_files[0])
    logger.info(f"Loading CSV file: {file_name}")

    try:
        data_frame = pd.read_csv(file_name)
        # Remove white space from column names
        data_frame.columns = [col.replace(" ", "") for col in data_frame.columns]
    except Exception as e:
        logger.error(f"Error reading CSV file {file_name}: {e}")
        return None

    return data_frame


def compute_model_metrics(y_true, y_pred, beta=1):
    """Compute precision, recall and F-beta score.

    Args:
        y_true (array-like): Actual labels.
        y_pred (array-like): Predicted labels.
        beta (float, optional): The beta parameter for F-beta score. Default is 1.

    Returns:
        tuple: precision, recall and F-beta score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=beta)

    return precision, recall, fbeta


def save_model_training_info(
    data: pd.DataFrame,
    best_model,
    initial_parameters,
    final_parameters,
    data_path: str,
    model_path: str,
    k_folds: int,
    params,
) -> None:
    data_shape = data.shape
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = f"""
    Model Training Information:

    1. Date and Time: {current_datetime}

    2. Data:
        - Shape: {data_shape}
        - Loaded from: {data_path}

    3. Model:
        - Type: {type(best_model).__name__}
        - Stored at: {model_path}

    4. Hyperparameters:
        - Initial: {json.dumps(initial_parameters, indent=12)}
        - Final (best): {json.dumps(final_parameters, indent=12)}

    5. Cross-Validation:
        - Type: K-Fold
        - Number of folds: {k_folds}

    6. Training Steps:
        - Split the data into features (X) and target (y)
        - Initialize the XGBoost model
        - Define the hyperparameters grid
        - Initialize a RandomizedSearchCV object for hyperparameter tuning
        - Fit the model and find the best parameters
        - Extract the best estimator
        - Initialize a KFold splitter
        - For each fold, fit the best model and calculate precision, recall, and F-beta scores
        - Save the best model to disk

     7. Metrics:
         {json.dumps(params, indent=12)}

     8. Metrics (Explained in the context of medical diagnostics):
        - Precision: This is the proportion of patients that your model correctly identified as having the disease out of all the patients it identified as having the disease. A higher precision means that when your model predicts a patient has the disease, it is highly likely that the patient truly has the disease. This is critical in scenarios where false positives could lead to unnecessary stress or treatments.

        - Recall (Sensitivity): This is the proportion of patients with the disease that your model correctly identifies. A higher recall means that your model is good at detecting the disease when it is present. This is crucial in conditions where early detection significantly improves outcomes.

        - F-beta score: This score is a balance between precision and recall. Depending on the clinical setting, you might want to emphasize recall (if missing a diagnosis has severe consequences) or precision (if false positives are particularly costly or harmful).

        - AUC (Area Under the ROC Curve): This metric tells you about the model's ability to distinguish between patients with the disease and without the disease. The AUC represents the probability that a randomly chosen positive (disease) example is ranked higher than a randomly chosen negative (no disease) example. The higher the AUC, the better the model is at differentiating between patients with the disease and without it.

        - Specificity: This is the proportion of healthy patients that your model correctly identified as being healthy. A higher specificity means that the model is good at avoiding false alarms in healthy patients. This is especially important in tests where a false positive could lead to additional invasive testing or unnecessary treatment.

    """

    report_dir = "../reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    file_path = os.path.join(report_dir, "model_training_report.txt")
    with open(file_path, "w") as file:
        file.write(info)

    print(f"Model training information saved to {file_path}")


def train_and_save_model(data, target, data_path: str = "../data"):
    """Train an XGBoost model and save it to disk.

    This function first prepares the data, then it initializes an XGBoost model and
    performs hyperparameter tuning using randomized search cross-validation. It uses
    k-fold cross-validation to calculate mean precision, recall, and F-beta scores.
    Finally, it saves the best model to disk.

    Args:
        data (pandas.DataFrame): The dataset.
        target (str): The name of the target column.
    """
    # Prepare the data
    X = data.drop(target, axis=1)
    y = data[target].replace(-1, 0)

    # Initialize an XGBoost model
    model = XGBClassifier(eval_metric="logloss")

    # Define the hyperparameters grid
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [6, 10, 15],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.5, 0.8, 1],
        "colsample_bytree": [0.5, 0.8, 1],
    }

    # Initialize a GridSearchCV object
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring="roc_auc",
    )

    # Fit the model and find best parameters
    random_search.fit(X, y)

    # Extract the best estimator
    best_model = random_search.best_estimator_
    # Initialize a KFold splitter
    kf = KFold(n_splits=5)

    # Lists to store each fold's scores
    (
        precision_scores,
        recall_scores,
        fbeta_scores,
        auc_scores,
        sensitivity_scores,
        specificity_scores,
    ) = ([], [], [], [], [], [])

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        pred_probs = best_model.predict_proba(X_test)[:, 1]

        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        auc = roc_auc_score(y_test, pred_probs)

        cm = confusion_matrix(y_test, preds)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        precision_scores.append(precision)
        recall_scores.append(recall)
        fbeta_scores.append(fbeta)
        auc_scores.append(auc)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)

    params = {}
    # Calculate the mean score across all folds
    params["mean_precision"] = np.mean(precision_scores)
    params["mean_recall"] = np.mean(recall_scores)
    params["mean_fbeta"] = np.mean(fbeta_scores)
    params["mean_auc"] = np.mean(auc_scores)
    params["mean_sensitivity"] = np.mean(sensitivity_scores)
    params["mean_specificity"] = np.mean(specificity_scores)

    # Save the model to disk
    path2models = os.path.join("..", "models")
    if not os.path.exists(path2models):
        os.makedirs(path2models)

    filename = os.path.join(path2models, "trained_model.joblib")
    dump(best_model, filename)

    # Generate model training report
    final_parameters = best_model.get_params()

    save_model_training_info(
        data, best_model, param_dist, final_parameters, data_path, filename, 5, params
    )


# %%
# =============================================================================
# MAIN CALLER
# =============================================================================


def main():
    """
    Main function that orchestrates the process of loading the data, training the model,
    saving the model, and generating a model report.

    The function does not take any arguments nor does it return anything. It performs the
    following steps:

    1. Load the data: This is done using the data_loader() function. The function looks for
       the first .csv file in the specified directory and loads it into a pandas DataFrame.

    2. Train and save the model: This is done using the train_and_save_model() function. The
       function trains a model on the loaded data with 'salary' as the target variable. It also
       performs a hyperparameter search and cross-validation, computes model metrics, saves
       the best model to disk, and generates a model report.

    Both functions log information about their progress, as well as any errors or warnings,
    to help with troubleshooting and understanding the flow of the function.

    """
    # load the data
    data = data_loader()

    # train and save the model (also generate the model report)
    train_and_save_model(data, "salary")


if __name__ == "__main__":
    # call the main function
    main()

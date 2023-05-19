"""
Check performance of model on slices of the categorical features of the data
"""

# Importing the necessary modules

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from path import data_folder_path
from data import process_data
from model_dev import compute_model_metrics
from model_dev import inference


def slices_scores():
    """
    Function that outputs the score of the model
    on slices of the categorical features of the data
    """

    # Loading the dataset
    df = pd.read_csv(data_folder_path("preprocessed", "preprocessed.csv"))

    # Splitting the data in train and test. We only need the test chunk
    # however and that is why we use the placeholder "_".
    _, test = train_test_split(df, test_size=0.20, random_state=20)

    # Loading the trained model
    trained_model = load(data_folder_path("model", "model.joblib"))
    encoder = load(data_folder_path("encoder", "encoder.joblib"))
    lb = load(data_folder_path("lb", "lb.joblib"))

    # Creating a list with the categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    #
    slices = []

    # We loop through the features we want and through each value of the
    # features, we process the data, predict and save the score metrics
    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            prediction = inference(trained_model, X_test)
            prc, rcl, fb = compute_model_metrics(y_test, prediction)

            # We save the score metrics and append each line to the
            # slices_scores list we created before
            line = f"|{cat}:{cls}| Precision: {prc} Recall: {rcl} FBeta: {fb}"
            slices.append(line)

            # We save the list in a .txt file
            with open(data_folder_path("slices", "slice_output.txt"), 'w') as out:  # noqa: E501
                for slice_value in slices:
                    out.write(slice_value + '\n')


if __name__ == "__main__":
    slices_scores()

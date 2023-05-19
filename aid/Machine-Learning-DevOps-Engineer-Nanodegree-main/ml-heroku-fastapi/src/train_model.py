"""
    Module to train and save the model
    """

# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from path import data_folder_path
from data import process_data
from model_dev import train_model

# Add code to load in the data.

data = pd.read_csv(data_folder_path("preprocessed", "preprocessed.csv"))
# train-test split.
train, _ = train_test_split(data, test_size=0.20, random_state=20)

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

# Proces the test data with the process_data function.

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
trained_model = train_model(X_train, y_train)
# dump(
#     trained_model,
#     "/home/antoniosf/code/Uda_MLOps_Project_3/data/model/model.joblib"
# )
# dump(
#     encoder,
#     "/home/antoniosf/code/Uda_MLOps_Project_3/data/model/encoder.joblib"
# )
# dump(
#     lb,
#     "/home/antoniosf/code/Uda_MLOps_Project_3/data/model/lb.joblib"
# )
dump(trained_model, data_folder_path("model", "model.joblib"))
dump(trained_model, data_folder_path("encoder", "encoder.joblib"))
dump(trained_model, data_folder_path("lb", "lb.joblib"))

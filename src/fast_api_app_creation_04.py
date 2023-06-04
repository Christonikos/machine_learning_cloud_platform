#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
from fastapi import FastAPI
from pydantic import BaseModel, Field
import argparse
import pandas as pd
import joblib
import logging
from typing import Optional, Any
from .model_slices_03 import load_model
from .data_injestion_01 import data_preprocessing

# =============================================================================
# UTILITY FUNCTIONS/CLASSES
# =============================================================================


app = FastAPI()

# Load the trained model
model = load_model()


def load_preprocessor(
    model_path: str = "../models/preprocessor.joblib",
) -> Optional[Any]:
    """
    Loads a trained machine learning model from a specified path.

    Parameters:
    model_path (str): The path (relative or absolute) to the .joblib file

    Returns:
    The trained preprocessor loaded into memory
    """
    try:
        # Load the model from the .joblib file
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"No such file or directory: '{model_path}'")
        return None
    except Exception as e:
        logging.error(f"Unexpected error occurred when loading the model: {e}")
        return None


class Data(BaseModel):
    age: int
    fnlgt: int
    workclass: str
    education: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int = Field(..., alias="hours-per-week")
    salary: str
    education_num: int = Field(..., alias="education-num")
    capital_gain: str = Field(..., alias="capital-gain")
    capital_loss: str = Field(..., alias="capital-loss")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 23,
                "fnlgt": 263886,
                "workclass": "Private",
                "education": "Some-college",
                "marital_status": "Never-married",
                "occupation": "Sales",
                "relationship": "Not-in-family",
                "race": "Black",
                "sex": "Female",
                "hours-per-week": 20,
                "native-country": "United-States",
                "salary": "<=50K",
                "education-num": 8,
                "capital-gain": 2174,
                "capital-loss": 0,
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to our Census Bureau Data Classifier"}


@app.post("/predict")
def predict(data: Data):
    """
    This function accepts a POST request with a JSON body containing the input data
    in a format corresponding to the Pydantic Data model. The function makes a prediction
    on the given data using a pre-trained model.

    Args:
        data (Data): A Pydantic model that validates the structure of the input data.

    Returns:
        A dictionary containing the prediction.
    """

    # Convert data to DataFrame
    # The `by_alias=True` argument is used because Pydantic models use
    # aliases for attribute names that aren't valid Python variable names.
    data_dict = data.dict(by_alias=True)
    data_df = pd.DataFrame(data_dict, index=[0])

    # Load the preprocessor
    # This preprocessor is a Scikit-learn ColumnTransformer that was used to preprocess the
    # training data. It is stored as a .joblib file and is loaded here for use.
    preprocessor = load_preprocessor()

    # Preprocess the data
    # The `data_preprocessing` function uses the loaded preprocessor to transform
    # the input data in the same way the training data was transformed.
    preprocessed_data = data_preprocessing(
        data_df, inference=True, preprocessor=preprocessor
    )

    # Ensure the order of columns matches the input of the trained model
    # The preprocessed data may have columns in a different order or missing entirely
    # compared to the training data. This step ensures the data is in the correct format.
    preprocessed_data = preprocessed_data.reindex(
        columns=model.feature_names_in_, fill_value=0
    )

    # Make prediction
    # The trained model is used to make a prediction on the preprocessed data.
    prediction = model.predict(preprocessed_data)

    # Return the prediction
    # The prediction is a binary outcome, where 0 represents '<=50K' and 1 represents '>50K'.
    # This step converts the numerical prediction back into a categorical form.
    if prediction[0] == 0:
        prediction = "<=50K"
    else:
        prediction = ">50K"

    # The prediction is returned as a JSON response.
    return {"prediction": prediction}


# %%
# =============================================================================
# MAIN CALLER
# =============================================================================


def main(args):
    """
    This function is the entry point of the script. It creates a sample data object and
    calls the `predict` function to make a prediction on this data.

    Args:
        args: The command-line arguments. This is not used in the function, but can be
              modified to allow input arguments for the script.
    """
    sample = {
        "age": 35,
        "fnlgt": 512345,
        "workclass": "Self-emp",
        "education": "Masters",
        "marital_status": "Married",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native_country": "Canada",
        "salary": "<=50K",
        "education-num": 8,
        "capital-gain": 2174,
        "capital-loss": 0,
    }
    data = Data(**sample)
    predict(data)


# =============================================================================
# WRAPER
# =============================================================================
if __name__ == "__main__":
    # =========================================================================
    # ARGPARSE INPUTS
    # =========================================================================
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-arg1", "--arg1_long", default="", help=" ")
    args = parser.parse_args()
    main(args)

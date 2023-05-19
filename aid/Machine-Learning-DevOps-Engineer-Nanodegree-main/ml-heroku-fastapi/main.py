"""
RESTful API using FastAPI with one GET on the root giving a welcome message
and one POST that does model inference
"""

import os
from typing import Literal
from pydantic import BaseModel
from fastapi import FastAPI
from joblib import load
from pandas.core.frame import DataFrame
import numpy as np
from src.data import process_data
from src.model_dev import inference


class ExampleToPredict(BaseModel):
    """
    Class to ingest the body from POST
    """
    age: int
    workclass: Literal['State-gov', 'Self-emp-not-inc', 'Private',
                       'Federal-gov', 'Local-gov', 'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                       'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate',
                       'Assoc-voc', 'Prof-school', '5th-6th', '10th',
                       'Preschool', '12th', '1st-4th']
    maritalStatus: Literal['Never-married', 'Married-civ-spouse', 'Divorced',
                           'Married-spouse-absent', 'Separated',
                           'Married-AF-spouse', 'Widowed']
    occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                        'Prof-specialty', 'Other-service', 'Sales',
                        'Transport-moving', 'Farming-fishing',
                        'Machine-op-inspct', 'Tech-support', 'Craft-repair',
                        'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
    relationship: Literal['Not-in-family', 'Husband', 'Wife', 'Own-child',
                          'Unmarried', 'Other-relative']
    race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
                  'Other']
    sex: Literal['Male', 'Female']
    hoursPerWeek: int
    nativeCountry: Literal['United-States', 'Cuba', 'Jamaica', 'India',
                           'Mexico', 'Puerto-Rico', 'Honduras', 'England',
                           'Canada', 'Germany', 'Iran', 'Philippines',
                           'Poland', 'Columbia', 'Cambodia', 'Thailand',
                           'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                           'Dominican-Republic', 'El-Salvador', 'France',
                           'Guatemala', 'Italy', 'China', 'South', 'Japan',
                           'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)',
                           'Scotland', 'Trinadad&Tobago', 'Greece',
                           'Nicaragua', 'Vietnam', 'Hong', 'Ireland',
                           'Hungary', 'Holand-Netherlands']


# Instantiate the app
app = FastAPI()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Define a GET on the specified andpoint
@app.get("/")
async def get_items():
    """Simple GET"""
    return {"greeting": "Hello there!"}


@app.post("/prediction")
async def make_inference(example_data: ExampleToPredict):
    """POST to return prediction from our saved model"""
    trained_model = load("data/model/model.joblib")
    encoder = load("data/encoder/encoder.joblib")
    lb = load("data/lb/lb.joblib")

    array = np.array([[
        example_data.age, example_data.workclass, example_data.fnlgt,
        example_data.education, example_data.maritalStatus,
        example_data.occupation, example_data.relationship, example_data.race,
        example_data.sex, example_data.hoursPerWeek, example_data.nativeCountry
    ]])

    df = DataFrame(data=array,
                   columns=[
                       "age",
                       "workclass",
                       "fnlgt",
                       "education",
                       "marital-status",
                       "occupation",
                       "relationship",
                       "race",
                       "sex",
                       "hours-per-week",
                       "native-country",
                   ])

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

    X, _, _, _ = process_data(df,
                              categorical_features=cat_features,
                              encoder=encoder,
                              lb=lb,
                              training=False)

    # prediction = trained_model.predict(X)
    prediction = inference(trained_model, X)
    y = lb.inverse_transform(prediction)[0]
    return {"prediction": y}

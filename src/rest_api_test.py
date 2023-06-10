#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    """
    Tests GET. Status code and if it is returning what is expected

    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Main API"}


def test_post_prediction(client):
    """
    Tests POST for a prediction.
    Status code and if the response has prediction field.

    """
    response = client.post(
        "/v1/predict",
        json={
            "age": 65,
            "fnlgt": 209280,
            "workclass": "State-gov",
            "education": "Masters",
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "hours-per-week": 35,
            "native-country": "United-States",
            "salary": "<=50K",
            "education-num": 8,
            "capital-gain": 2174,
            "capital-loss": 0,
        },
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_post_less_than_50(client):
    """
    Tests POST for a prediction less than 50k.
    Status code and if the prediction is the expected one

    """
    response = client.post(
        "/v1/predict",
        json={
            "age": 23,
            "fnlgt": 263886,
            "workclass": "Private",
            "education": "Some-college",
            "marital-status": "Never-married",
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
        },
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

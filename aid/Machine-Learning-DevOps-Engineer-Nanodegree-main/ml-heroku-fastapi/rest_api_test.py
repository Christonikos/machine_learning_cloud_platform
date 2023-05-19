"""
This is unit test for rest_api.py
"""

import pytest
from fastapi.testclient import TestClient
from main import app


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
    assert response.json() == {"greeting": "Hello there!"}


def test_post_more_than_50(client):
    """
    Tests POST for a prediction less than 50k.
    Status code and if the prediction is the expected one

    """
    response = client.post("/prediction",
                           json={
                               "age": 65,
                               "fnlgt": 209280,
                               "workclass": "State-gov",
                               "education": "Masters",
                               "maritalStatus": "Married-civ-spouse",
                               "occupation": "Prof-specialty",
                               "relationship": "Husband",
                               "race": "White",
                               "sex": "Male",
                               "hoursPerWeek": 35,
                               "nativeCountry": "United-States"
                           })

    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_post_less_than_50(client):
    """
    Tests POST for a prediction more than 50k.
    Status code and if the prediction is the expected one

    """
    response = client.post("/prediction",
                           json={
                               "age": 23,
                               "fnlgt": 263886,
                               "workclass": "Private",
                               "education": "Some-college",
                               "maritalStatus": "Never-married",
                               "occupation": "Sales",
                               "relationship": "Not-in-family",
                               "race": "Black",
                               "sex": "Female",
                               "hoursPerWeek": 20,
                               "nativeCountry": "United-States"
                           })

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

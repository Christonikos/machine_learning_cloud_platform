#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
import requests

# The URL of the live Render API.
url = "https://census-ml-api-n83d.onrender.com"

payload = {
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
}

# Make the POST request and store the response.
response = requests.post(url, json=payload)

# If the POST request was successful, the status code will be 200.
if response.status_code == 200:
    print("POST request was successful.")
    print(f"Response: {response.json()}")
else:
    print(f"POST request failed with status code {response.status_code}.")

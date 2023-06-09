#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
import requests
import json

# Define the URL
url = "https://census-ml-api-n83d.onrender.com/predict"

# Define the data to be sent to the API
data = {
    "age": 35,
    "fnlgt": 512345,
    "workclass": "Self-emp",
    "education": "Masters",
    "marital_status": "Married",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hours_per_week": 40,
    "native_country": "Canada",
    "salary": "<=50K",
    "education_num": 8,
    "capital_gain": 2174,
    "capital_loss": 0,
}

# Convert data to json format
json_data = json.dumps(data)

# Send the request to the API
response = requests.post(url, json=json_data)

# Print the response
print(response.json())

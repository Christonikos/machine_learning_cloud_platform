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

# define the API endpoint
url = "http://0.0.0.0:10000/v1/predict"

# define the data to be sent to the API
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
    "hours-per-week": 40,
    "native_country": "Canada",
    "salary": "<=50K",
    "education-num": 8,
    "capital-gain": 2174,
    "capital-loss": 0,
}

# convert dict to json
data_json = json.dumps(data)

# set the headers
headers = {"Content-Type": "application/json"}

# make the POST request
response = requests.post(url, data=data_json, headers=headers)

# print the response status code and JSON
print(f"Status code: {response.status_code}")
print(f"Response JSON: {response.json()}")

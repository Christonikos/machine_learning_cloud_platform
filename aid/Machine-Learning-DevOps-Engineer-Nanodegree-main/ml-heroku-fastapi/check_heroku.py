""" Module to test heroku deployment"""

import requests

data = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 65278,
    "education": "HS-grad",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
}

r = requests.post('https://antonios-ml-project-app.herokuapp.com/prediction',
                  json=data)

assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")

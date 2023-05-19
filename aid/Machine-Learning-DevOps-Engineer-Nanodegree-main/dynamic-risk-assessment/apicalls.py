import requests
import json
import os


# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)
model_path = os.path.join(config['output_model_path'])

# Call each API endpoint and store the responses
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
response1 = requests.post(
    f'{URL}/prediction?inputdata=testdata/testdata.csv', headers=headers).text

response2 = requests.get(f'{URL}/scoring', headers=headers).text
response3 = requests.get(f'{URL}/summarystats',
                         headers=headers).text
response4 = requests.get(f'{URL}/diagnostics',
                         headers=headers).text

# combine all API responses
responses = str(response1 + response2 + response3 +
                response4)

# write the responses to your workspace
with open(os.path.join(model_path, "apireturns.txt"), "w") as api_returns:
    api_returns.write(responses)

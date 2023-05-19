
import subprocess
from posix import times_result
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import os
import json
from joblib import load
import pandas as pd

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']

output_model_path = os.path.join(config['output_model_path'])


# Check and read new data
# first, read ingestedfiles.txt

ingested_files = []
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as report_file:
    for line in report_file:
        ingested_files.append(line.rstrip())
# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
files = os.listdir(input_folder_path)
new_files = False
for filename in files:
    if filename not in ingested_files:
        new_files = True

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    print('No new data were found, exiting')
    exit()


# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as report_file:
    old_f1 = float(report_file.read())

ingestion.merge_multiple_dataframe()
final_data = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
ML_model = load(os.path.join(prod_deployment_path, "trainedmodel.pkl"))
scoring.score_model(final_data, ML_model)

with open(os.path.join(output_model_path, "latestscore.txt"), "r") as report_file:
    new_f1 = float(report_file.read())
# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here

# Because running the script, no drift occurs I manully set a low F1 score to test my script
# new_f1 = 0.5
if new_f1 >= old_f1:
    print(f'New f1 score {new_f1} does not indicate model drift, exiting')
    exit()

print(
    f'New f1 score {new_f1} indicates model drift. Starting model retraining')
training.train_model()

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
model = str(os.path.join(os.getcwd(), output_model_path, "trainedmodel.pkl"))

deployment.store_model_into_pickle(model)
##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
# diagnostics.model_predictions(final_data)
# reporting.score_model()
# diagnostics.dataframe_summary()
# diagnostics.missing_values()
# diagnostics.execution_time()
# diagnostics.outdated_packages_list()

# Run reporting.py, apicalls.py
subprocess.run(['python', 'reporting.py'])

subprocess.run(['python', 'apicalls.py'])

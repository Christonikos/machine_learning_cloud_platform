import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    extension = 'csv'
    os.chdir(input_folder_path)
    # csv_files = glob.glob('*.{}'.format(extension))
    csv_files = glob.glob(f'*.{extension}')

    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
    df.drop_duplicates(inplace=True)

    os.chdir(os.path.dirname(os.getcwd()))

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    os.chdir(output_folder_path)
    df.to_csv('finaldata.csv', index=False)

    with open('ingestedfiles.txt', 'w') as txt_file:
        for file in csv_files:
            txt_file.write(file + '\n')


if __name__ == '__main__':
    merge_multiple_dataframe()

"""
This script ingests data from the input folder, removes duplicates and saves the result to the output folder.

Usage: python ingestion.py
Author: P. Arvanitis
"""
import pandas as pd
import os
import json
from datetime import datetime


# Read config file
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

def find_data_files(root_dirs):
    """
    Find all data files (csv, json, xlsx) in the root directories.

    :param root_dirs: root directories
    :type root_dirs: list
    :return: data files
    :rtype: list
    """
    data_files = []
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.json') or file.endswith('.xlsx'):
                    data_files.append(os.path.join(root, file))
    return data_files


def read_data_files(data_files):
    """
    Read data files into a list of pandas dataframes

    :param data_files: data files
    :type data_files: list
    :return: pandas dataframes
    :rtype: list
    """
    data_frames = []
    for data_file in data_files:
        if data_file.endswith('.csv'):
            data_frames.append(pd.read_csv(data_file))
        elif data_file.endswith('.json'):
            data_frames.append(pd.read_json(data_file))
        elif data_file.endswith('.xlsx'):
            data_frames.append(pd.read_excel(data_file))
    return data_frames

def compile_dataframes(data_frames):
    """
    Compile a list of pandas dataframes into a single dataframe

    :param data_frames: pandas dataframes
    :type data_frames: list
    :return: a single dataframe
    :rtype: pandas.DataFrame
    """
    return pd.concat(data_frames, ignore_index=True)

def remove_duplicates(df):
    """
    Remove duplicate rows from a dataframe

    :param df: a dataframe
    :type df: pandas.DataFrame
    :return: a dataframe without duplicate rows
    :rtype: pandas.DataFrame
    """
    return df.drop_duplicates()

def save_to_csv(df):
    """
    Save a dataframe to a csv file

    :param df: a dataframe
    :type df: pandas.DataFrame
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    df.to_csv(os.path.join(output_folder_path,'finaldata.csv'), index=False)


def ingest_data():
    """
    Ingest data from the input folder, remove duplicates and save the result to the output folder.
    Return the list of data files that were ingested.

    :return: data files
    :rtype: list
    """
    print('Ingesting data...')

    input_folders = [input_folder_path]
    data_files = find_data_files(input_folders)
    data_frames = read_data_files(data_files)
    df = compile_dataframes(data_frames)
    df = remove_duplicates(df)
    save_to_csv(df)

    print('Data ingestion complete.')

    return data_files

def save_data_record(data_files, output_file_name: str = 'ingestedfiles.txt'):
    """
    Save a record of the data ingestion process to a txt file in the output folder, that is a list of the
    data file names that were ingested.

    :param data_files: data files
    :type data_files: list
    :param output_file_name: output file name
    :type output_file_name: str
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Sort data files by name
    data_files.sort()

    # Get current date and time
    now = datetime.now()

    # Save data ingestion record
    with open(os.path.join(output_folder_path, output_file_name), 'w') as f:
        for data_file in data_files:
            f.write(data_file + '\n')
            # f.write(os.path.basename(data_file) + '\n')
        f.write('\n')
        f.write('Data ingestion complete at ' + now.strftime("%d/%m/%Y %H:%M:%S"))


    print('Data ingestion record saved.')


if __name__ == '__main__':
    data_files_ingested = ingest_data()
    save_data_record(data_files_ingested)

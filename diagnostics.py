import pickle

import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def get_model_predictions(test_df):
    """
    Score the model on the test data
    :param test_df: test data
    :type test_df: pandas.DataFrame
    :return: predictions
    :rtype: list
    """
    # Load the trained model from the production deployment folder
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    # Discard 'corporation' column
    test_df = test_df.drop(columns=['corporation'], axis=1)

    # Get X and y (y is the 'exited' column)
    X = test_df.drop(columns=['exited'], axis=1)
    y = test_df['exited']

    # Calculate predictions
    predictions = model.predict(X)

    # Return predictions
    return predictions.tolist()


def get_dataframe_summary():
    """
    Calculate summary statistics (means, medians, stds, etc) for each numeric column of the final dataset and
    return them as a list
    :return: summary statistics
    :rtype: list
    """
    # Read the dataset from the output folder
    df = pd.read_csv(dataset_csv_path)

    # Calculate summary statistics only for numeric columns
    summary = df.describe()

    # Calculate median
    median = df.median()

    # Add median to summary statistics
    summary.loc['median'] = median

    # Convert the summary statistics to a list
    summary_list = []

    for column in summary:
        summary_list.append(summary[column])

    return summary_list


def check_for_missing_values():
    """
    Check the final dataset for missing values and return a list with what percentage of values are missing
     for each column
    :return: missing values percentages
    :rtype: list
    """
    # Read the dataset from the output folder
    df = pd.read_csv(dataset_csv_path)

    # Calculate the percentage of missing values for each column
    missing_values_percentages = df.isnull().mean() * 100

    # Convert the percentages to a list
    missing_values_percentages_list = []

    for column in df.columns:
        missing_values_percentages_list.append(missing_values_percentages[column])

    return missing_values_percentages_list


def get_execution_times():
    """
    Calculate the time it takes to run training.py and ingestion.py and return them as a list
    :return: timing values
    :rtype: list
    """
    # Ingestion timing
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time

    # Training timing
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time

    # Return the timing values as a list
    return [training_time, ingestion_time]


def get_outdated_packages():
    """
    Get a table with three columns: package name, installed version, and latest version of all the outdated packages
    installed using pip
    :return: outdated packages
    :rtype: numpy.ndarray
    """
    # Get outdated packages
    os.system('pip list --outdated --format=json > outdated_packages.json')
    outdated_packages = pd.read_json('outdated_packages.json')

    # Return the outdated packages as a numpy array
    return outdated_packages.values


if __name__ == '__main__':
    # Get test data
    test_data = pd.read_csv(test_data_path)

    # Get model predictions
    predictions = get_model_predictions(test_data)

    # Get summary statistics
    summary = get_dataframe_summary()

    # Get missing values percentages
    missing_values_percentages = check_for_missing_values()

    # Get execution times list (training, ingestion)
    execution_times = get_execution_times()

    # Get packages info
    outdated_packages = get_outdated_packages()

    # Print results
    print('Model predictions: {}'.format(predictions))
    print('Summary statistics: {}'.format(summary))
    print('Missing values percentages: {}'.format(missing_values_percentages))
    print('Execution times: {}'.format(execution_times))
    print('Outdated packages: {}'.format(outdated_packages))

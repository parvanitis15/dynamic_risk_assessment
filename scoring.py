from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
output_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def score_model(model):
    """
    Score the model on the test data

    :param model: trained model
    :type model: sklearn.linear_model.LogisticRegression
    :return: F1 score
    :rtype: float
    """
    # Read the test data
    df = pd.read_csv(test_data_path)

    # Discard 'corporation' column
    df = df.drop(columns=['corporation'], axis=1)

    # Get X and y (y is the 'exited' column)
    X = df.drop(columns=['exited'], axis=1)
    y = df['exited']

    # Calculate F1 score
    score = model.score(X, y)

    # Write the result to the latestscore.txt file
    with open('latestscore.txt', 'w') as f:
        f.write(str(score))

    return score


if __name__ == '__main__':
    # Load the model from the output_model_path
    model = pickle.load(open(output_model_path, 'rb'))

    # Calculate the score and print it to the console
    score = score_model(model)
    print("F1 score: %.2f%%" % (score * 100.0))

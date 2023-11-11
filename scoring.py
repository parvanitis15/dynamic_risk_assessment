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


def score_model():
    """
    Score the model on the test data
    """
    # Read the test data
    df = pd.read_csv(test_data_path)

    # Load the trained model
    model = pickle.load(open(output_model_path, 'rb'))

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


if __name__ == '__main__':
    score_model()

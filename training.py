from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


def train_model(test_size=0.2, random_state=42):
    """
    Train the model on the training data
    :param test_size: test size for train_test_split
    :type test_size: float
    :param random_state: random state for train_test_split
    :type random_state: int
    """

    # Read the dataset from the specified path
    df = pd.read_csv(dataset_csv_path)

    # Discard 'corporation' column
    df = df.drop(columns=['corporation'], axis=1)

    # Get X and y (y is the 'exited' column)
    X = df.drop(columns=['exited'], axis=1)
    y = df['exited']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # Fit the logistic regression to your data
    # model.fit(X_train, y_train)
    model.fit(X, y)

    # Write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(model_path, 'wb'))

    # Print the accuracy of the model
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

    # Print the confusion matrix
    y_pred = model.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_pred))

    # Print the precision, recall, and f1-score
    print(metrics.classification_report(y_test, y_pred))


if __name__ == '__main__':
    train_model()

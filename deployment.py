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
output_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])


def copy_to_production():
    """
    Copy the model file and latestscore.txt to the production deployment folder
    """
    # Copy model file to deployment directory
    os.system('cp {} {}'.format(output_model_path, os.path.join(prod_deployment_path, 'trainedmodel.pkl')))

    # Copy latestscore.txt to deployment directory
    os.system('cp latestscore.txt {}'.format(os.path.join(prod_deployment_path, 'latestscore.txt')))

    # Copy ingestfiles.txt to deployment directory
    os.system('cp {} {}'.format(os.path.join(output_folder_path, 'ingestedfiles.txt') ,os.path.join(prod_deployment_path, 'ingestedfiles.txt')))

        
if __name__ == '__main__':
    copy_to_production()

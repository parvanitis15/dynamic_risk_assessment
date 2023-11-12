import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import get_model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')


##############Function for reporting
def generate_confusion_matrix(predictions: list, actual_values: list, output_file_name: str = 'confusionmatrix.png'):
    """
    Generate a confusion matrix from the predictions and save to confusion matrix plot to the workspace folder
    :param predictions: predictions from the model
    :type predictions: list
    :param actual_values: actual values from the test data
    :type actual_values: list
    :param output_file_name: name of the confusion matrix plot file
    :type output_file_name: str
    """
    cm = metrics.confusion_matrix(actual_values, predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(metrics.accuracy_score(actual_values, predictions))
    plt.title(all_sample_title, size = 15)
    plt.savefig(os.path.join(output_model_path, output_file_name))
    plt.show()


if __name__ == '__main__':
    # Get model predictions on the test data
    test_df = pd.read_csv(test_data_path)

    # Load the model from the output_model_path
    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb'))

    predictions = get_model_predictions(test_df, model)

    # Get actual values from the test data
    actual_values = test_df['exited'].values.tolist()

    # Generate confusion matrix
    generate_confusion_matrix(predictions, actual_values)


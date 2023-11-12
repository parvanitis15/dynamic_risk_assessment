from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import get_model_predictions, get_dataframe_summary, check_for_missing_values, get_execution_times, \
    get_outdated_packages
from scoring import score_model
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = pickle.load(open(os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl'), 'rb'))


@app.route('/prediction', methods=['POST'])
def predict():
    """
    Call this api to make predictions on new data using the deployed model and return the predictions.

    :return: predictions
    :rtype: list
    """
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is present and has an allowed extension
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Assuming only CSV files are allowed
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'})

    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV file: {str(e)}'})

    # Make predictions using the get_model_predictions function
    try:
        model = prediction_model # Load your pre-loaded model here
        predictions = get_model_predictions(df, model)
    except Exception as e:
        return jsonify({'error': f'Error generating predictions: {str(e)}'})

    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions})


@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """
    Call this api to get the latest F1 score of the deployed model and the test data.

    :return: F1 score
    :rtype: float
    """
    # Load the model
    try:
        model = prediction_model # Load your pre-loaded model here
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'})

    # Calculate the F1 score using the score_model function
    try:
        score = score_model(model)
    except Exception as e:
        return jsonify({'error': f'Error calculating F1 score: {str(e)}'})

    # Return the F1 score as a JSON response
    return jsonify({'F1 score': score})


@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    """
    Call this api to get summary statistics on the final dataset.
    Hint: Use the get_dataframe_summary function from diagnostics.py

    :return: summary statistics
    :rtype: list
    """
    # Get summary statistics using the get_dataframe_summary function
    try:
        summary = get_dataframe_summary()
    except Exception as e:
        return jsonify({'error': f'Error calculating summary statistics: {str(e)}'})

    # Summary is a list of pandas.Series objects. Convert it to a list of dictionaries
    summary_list = []
    for series in summary:
        summary_list.append(series.to_dict())

    # Return the summary statistics as a JSON response
    return jsonify({'summary': summary_list})

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    Call this api to get diagnostics (missing values, execution times, and outdated packages) on the final dataset.
    """
    # Get missing values percentages using the check_for_missing_values function
    try:
        missing_values_percentages = check_for_missing_values()
    except Exception as e:
        return jsonify({'error': f'Error calculating missing values percentages: {str(e)}'})

    # Get execution times using the get_execution_times function
    try:
        execution_times = get_execution_times()
    except Exception as e:
        return jsonify({'error': f'Error calculating execution times: {str(e)}'})

    # Get outdated packages using the get_outdated_packages function
    try:
        outdated_packages = get_outdated_packages()
    except Exception as e:
        return jsonify({'error': f'Error calculating outdated packages: {str(e)}'})

    # Return the diagnostics as a JSON response
    return jsonify({
        'missing_values_percentages': missing_values_percentages,
        'execution_times': execution_times,
        'outdated_packages': outdated_packages.tolist()
    })

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

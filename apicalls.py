import json
import os

import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:5000/"

#Load config.json and get the path variables
with open('config.json','r') as f:
    config = json.load(f)
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

def call_api_endpoints(output_file_name: str = 'apireturns.txt'):

    # Call each of the 4 API endpoints and store the responses

    # The predict endpoint is a POST request
    response1 = requests.post(URL + 'prediction', files={'file': open(test_data_path, 'rb')})

    # The rest use get requests
    response2 = requests.get(URL + 'scoring')
    response3 = requests.get(URL + 'summarystats')
    response4 = requests.get(URL + 'diagnostics')

    # Combine all API responses
    responses = [response1, response2, response3, response4]

    # Write the responses to the output folder
    with open(os.path.join(output_model_path, output_file_name), 'w') as f:
        for response in responses:
            f.write(response.text + '\n')


if __name__ == '__main__':
    call_api_endpoints()


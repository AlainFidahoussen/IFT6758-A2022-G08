"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import comet_ml
import pickle

import sys
sys.path.append('../')
import ift6758

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

model = None
model_path = "../models"

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # Setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # Read the log file specified and return the data
    with open("flask.log") as f:
        response = f.readlines()

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    api = comet_ml.api.API(api_key=os.environ.get('COMET_API_KEY'))

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Check to see if the model you are querying for is already downloaded
    workspace = json['workspace']
    model_name = json['model']
    version = json['version']

    # Convert model name to find model file
    try:
        # app.logger.info(api.get_registry_model_details(workspace, model_name, version)["assets"])
        filename = api.get_registry_model_details(workspace, model_name, version)["assets"][0]["fileName"]
    except:
        app.logger.info(f"Could not find {model_name}.")
        return ('', 401)
    is_downloaded = filename in os.listdir("../models")
    global model
    
    # If yes, load that model and write to the log about the model change.  
    if is_downloaded:
        with open(os.path.join(model_path, filename), 'rb') as f:
            model = pickle.load(f)
        app.logger.info(f"Loaded {model_name} (already downloaded).")
    else:
        # If no, try downloading the model: if it succeeds, load that model and write to the log about the model change. If it fails, write to the log about the failure and keep the currently loaded model.
        try:
            api.download_registry_model(workspace, model_name, version, output_path=model_path)
            app.logger.info(f"Downloaded {filename}.")
            with open(os.path.join(model_path, filename), 'rb') as f:
                model = pickle.load(f)
            app.logger.info(f"Loaded {model_name}.")
        except:
            app.logger.info(f"Failed to download model {model_name}, keeping current model.")
    app.logger.info(str(model))
    return ('', 204)
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    if model == None:
        # No model has been loaded yet
        return "No model loaded!", 403

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    pd.read_json(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

# Custom routes for testing
@app.route("/")
def default():
    """Testing if Flask works."""
    """Start server with gunicorn --bind 0.0.0.0:6758 app:app"""
    """To check this page go to http://127.0.0.1:6758/"""
    app.logger.info("Hello World!")
    return "Hello World"

@app.route("/test_download")
def test_download():
    workspace = "ift6758-a22-g08"
    model_name = "xgboost-randomforest"
    version = "1.0.0"
    import json
    import requests
    requests.post("http://127.0.0.1:6758/download_registry_model", json={
        "workspace": workspace,
        "model": model_name,
        "version": version
    })
    return ""
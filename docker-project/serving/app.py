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
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    # Maybe download baseline model?

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    raise NotImplementedError("TODO: implement this endpoint")

    response = None
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
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Check to see if the model you are querying for is already downloaded
    workspace = json['workspace']
    model_name = json['model']
    version = json['version']
    is_downloaded = model_name in os.listdir("../models")
    
    # If yes, load that model and write to the log about the model change.  
    api = comet_ml.api.API(api_key=os.environ.get('COMET_API_KEY'))
    if is_downloaded:
        app.logger.info(f"Loaded {model_name} (already downloaded).")
    else:
        # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log about the model change. If it fails, write to the log about the failure and keep the currently loaded model.
        try:
            api.download_registry_model(workspace, model_name, version, output_path=model_path)
            app.logger.info(f"Downloaded {model_name}.")
            # TODO: load model & log it
            with open(os.path.join(model_path, "XGBoost_SelectKBest.pkl"), 'rb') as f:
                global model
                model = pickle.load(f)
            app.logger.info(f"Loaded {model_name}.")
        except:
            app.logger.info(f"Failed to download model {model_name}, keeping current model.")
    
    

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    raise NotImplementedError("TODO: implement this endpoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

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
    return "Hello World"

@app.route("/test_download")
def test_download():
    api = comet_ml.api.API(api_key=os.environ.get('COMET_API_KEY'))
    workspace = "ift6758-a22-g08"
    model_name = "xgboost-randomforest"
    version = "1.0.0"
    # return api.get_registry_model_names(workspace)
    api.download_registry_model(workspace, model_name, version, output_path=model_path)
    return f"Downloaded {model_name}!"

@app.route("/downloaded")
def check_downloaded_model():
    pass
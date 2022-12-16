import os
import logging
from flask import Flask, jsonify, request, abort
import pandas as pd
import numpy as np
import comet_ml
import cloudpickle as pickle
# import pickle5 as pickle
import json

app = Flask(__name__)

cache = {}

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_DIR = os.environ.get("MODEL_DIR", "./comet_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# http://0.0.0.0:5555/predict


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # Setup logger
    format = "%(asctime)s;%(levelname)s;%(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=format)


@app.route("/logs", methods=["GET"])
def get_logs():
    with open(LOG_FILE) as f:
        data = f.read().splitlines()
    return jsonify(data)

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

    if 'COMET_API_KEY' not in os.environ.keys():
        app.logger.error(f"Please define the COMET_API_KEY.")
        return ('', 401)

    api = comet_ml.api.API(api_key=os.environ.get('COMET_API_KEY'))

    # Get POST json data
    data = json.loads(request.get_json())

    # Check to see if the model you are querying for is already downloaded
    workspace = data['workspace']
    model_name = data['model']
    version = data['version']

    app.logger.info("Request = " + str(data))


    # Convert model name to find model file
    try:
        # app.logger.info(api.get_registry_model_details(workspace, model_name, version)["assets"])
        filename = api.get_registry_model_details(workspace, model_name, version)["assets"][0]["fileName"]
    except:
        app.logger.error(f"Could not find {model_name}.")
        return ('', 401)

    is_downloaded = filename in os.listdir(MODEL_DIR)
    
    # If yes, load that model and write to the log about the model change.  
    if is_downloaded:
        with open(os.path.join(MODEL_DIR, filename), 'rb') as f:
            cache['model'] = pickle.load(f)
        app.logger.info(f"Loaded {model_name} (already downloaded).")
    else:
        # If no, try downloading the model: if it succeeds, load that model and write to the log about the model change. If it fails, write to the log about the failure and keep the currently loaded model.
        try:
            api.download_registry_model(workspace, model_name, version, output_path=MODEL_DIR)
            app.logger.info(f"Downloaded {filename}.")
            with open(os.path.join(MODEL_DIR, filename), 'rb') as f:
                cache['model'] = pickle.load(f)
            app.logger.info(f"Loaded {model_name}.")
        except Exception as e:
            app.logger.error(f"Failed to download model {model_name}, keeping current model.")
            app.logger.error(e)

    # app.logger.info(str(model)) # error because the pipelines of the model registry do not have a str function
    return ('', 204)
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here



@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """

    if 'model' not in cache.keys():
        # No model has been loaded yet
        app.logger.error(f"No model loaded!")
        return jsonify([])

    # Get POST json data
    json = request.get_json()

    X = pd.DataFrame.from_dict(json)
    app.logger.info("DataFrame Shape = " + str(X.shape))
    if X.shape[0] == 0:
        app.logger.warning("Empty DataFrame!")
        response = []

    else:
        try:
            preds = cache['model'].predict_proba(X)[:,1]
            response = preds.tolist()
            app.logger.info("Shot probability Sum = " + str(np.array(response).sum()))
        except Exception as e:
            app.logger.error("Failed to predict")
            app.logger.error(e)
            response = []

    return jsonify(response)  # response must be json serializable!



@app.route("/")
def default():
    """Testing if Flask works."""
    """Start server with gunicorn --bind 0.0.0.0:5000 app:app"""
    """To check this page go to http://127.0.0.1:5000/"""
    return open(LOG_FILE).readlines()

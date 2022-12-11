from dotenv import load_dotenv
load_dotenv();

import os
import sys
sys.path.insert(0, os.path.join(os.environ['NHL_DATA_DIR'], '..'))

import json
import requests
import pandas as pd
import logging
import src.features.build_features as FeaturesManager



logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 6758):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")


        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        # Remove the ground truth
        X_features = X.drop(labels=['Is Goal'], axis=1)

        # Send the request to the server to get the prediction (goal probability)
        request = requests.post(
            url = self.base_url + '/predict', 
            json = json.loads(X_features.to_json()))

        # Get the response, in json format
        response = request.json() # Not sure, we should test if we actually got a valid answer!

        if len(response) == X_features.shape[0]:
            # Append the response to the dataframe
            X_out = X.copy()
            X_out['Shot probability'] = response
            return X_out

        else:
            return X

    def logs(self) -> dict:
        """Get server logs"""

        request = requests.get(
            url = self.base_url + '/logs')

        response = request.json()

        response = ''.join(response)
        return response


    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """


        request_dict = {"workspace": workspace, "model": model, "version": version}

        # Send the request to the server to download the model
        _ = requests.post(
            url = self.base_url + '/download_registry_model', 
            json = json.dumps(request_dict))

        return {}


if __name__ == "__main__":

    sc = ServingClient()

    # Send a download request
    workspace = "ift6758-a22-g08"
    model = "randomforest-allfeatures"
    version = "1.0.0"
    # print(sc.download_registry_model(workspace, model, version))

    # Get the logs
    print(sc.logs())

    season_year = 2018
    season_type = "Regular"
    game_number = 20
    features_df = FeaturesManager.build_features_one_game(season_year, season_type, game_number, with_player_stats=True, with_strength_stats=True)

    print(sc.predict(features_df))
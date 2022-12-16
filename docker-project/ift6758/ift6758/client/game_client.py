import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import serving_client
import pandas as pd

import features.build_features as FeaturesManager
import visualization.visualize as VizManager

IP = os.environ.get("SERVING_IP", "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", "8890")

class GameClient:
    def __init__(self):
        self.tracker = []
        self.logs = ""


    def ping_game(self, season_year: str, season_type: str, game_number: str) -> pd.DataFrame:

        df_features = FeaturesManager.build_features_one_game(
            season_year=season_year, 
            season_type=season_type, 
            game_number=game_number, 
            with_player_stats=True, 
            with_strength_stats=True)

        if df_features is None:
            return pd.DataFrame()

        # Get only the new events (not in the tracker)
        df_features = df_features.loc[~df_features['Event Index'].isin(self.tracker)]

        # Update the tracker
        list_events = df_features['Event Index']
        self.tracker.extend(list_events)

        return df_features


    def get_heat_map(self, df_features: pd.DataFrame):

        teams = df_features['Team'].unique()

        [x_shots_0, y_shots_0] = VizManager.get_shots_coordinates(df_features, teams[0])
        [x_shots_1, y_shots_1] = VizManager.get_shots_coordinates(df_features, teams[1])

        shots_hist2D_0, _, _ = VizManager.get_shots_hist2D(x_shots_0, y_shots_0)
        shots_hist2D_1, _, _ = VizManager.get_shots_hist2D(x_shots_1, y_shots_1)

        return {teams[0]: shots_hist2D_0, teams[0]: shots_hist2D_1}

        

if __name__ == "__main__":

    # Load a model
    sc = serving_client.ServingClient(IP, PORT)
    workspace = "ift6758-a22-g08"
    model = "randomforest-allfeatures"
    version = "1.0.0"
    sc.download_registry_model(workspace, model, version)

    # Get a prediction
    gc = GameClient()
    season_year = 2016
    season_type = "Regular"
    game_number = 20
    df_features = gc.ping_game(season_year, season_type, game_number)

    df_features_out = sc.predict(df_features)

    # Should not process the events the second time, as it already did before
    df_features = gc.ping_game(season_year, season_type, game_number)
    df_features_out = sc.predict(df_features)


    # Try another model
    model = "xgboost-randomforest-ii"
    sc.download_registry_model(workspace, model, version)

    # Get a prediction
    game_number = 25
    df_features = gc.ping_game(season_year, season_type, game_number)
    df_features_out = sc.predict(df_features)

    print(sc.logs())
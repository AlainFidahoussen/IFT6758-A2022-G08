from dotenv import load_dotenv
load_dotenv();


import serving_client
import pandas as pd
from collections import defaultdict
import src.features.build_features as FeaturesManager
import src.data.NHLDataManager as DataManager

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

        # Get only the new events (not in the tracker)
        df_features = df_features.loc[~df_features['Event Index'].isin(self.tracker)]

        # Update the tracker
        list_events = df_features['Event Index']
        self.tracker.extend(list_events)

        return df_features


if __name__ == "__main__":

    sc = serving_client.ServingClient()
    workspace = "ift6758-a22-g08"
    model = "randomforest-allfeatures"
    version = "1.0.0"
    sc.download_registry_model(workspace, model, version)

    gc = GameClient()
    season_year = 2016
    season_type = "Regular"
    game_number = 20
    df_features = gc.ping_game(season_year, season_type, game_number)
    df_features_out = sc.predict(df_features)

    df_features = gc.ping_game(season_year, season_type, game_number)
    if len(df_features > 0):
        df_features_out = sc.predict(df_features)

    print(sc.logs())
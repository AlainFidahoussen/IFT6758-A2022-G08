from dotenv import load_dotenv
load_dotenv();


import serving_client
import pandas as pd
from collections import defaultdict
import src.features.build_features as FeaturesManager
import src.data.NHLDataManager as DataManager

class GameClient:
    def __init__(self):
        self.tracker = defaultdict(list)
        self.service_client = serving_client.ServingClient()
        self.logs = ""


    def get_game_features(self, season_year: str, season_type: str, game_number: str) -> pd.DataFrame:

        df_features = FeaturesManager.build_features_one_game(
            season_year=season_year, 
            season_type=season_type, 
            game_number=game_number, 
            with_player_stats=True, 
            with_strength_stats=True)

        if df_features is None:
            self.logs = f"Failed to get features data from season {season_year}/{season_type}, game {game_number}"
        else:
            self.logs = f"Succeed to get features data from season {season_year}/{season_type}, game {game_number}"

        return df_features


    def get_game_prediction(self, season_year: str, season_type: str, game_number: str):

        df_features = self.get_game_features(
            season_year=season_year,
            season_type=season_type,
            game_number=game_number)


        df_features_out = self.service_client.predict(df_features)

        if len(df_features_out) > 0:

            data_manager = DataManager.NHLDataManager()
            game_id = data_manager.get_game_id(
                season_year=season_year,
                season_type=season_type,
                game_number=game_number)

            # Get the list of events we just processed and update the tracker, so we don't do it again
            list_events = df_features_out['Event Index']
            self.tracker[game_id].extend(list_events)

        return df_features_out
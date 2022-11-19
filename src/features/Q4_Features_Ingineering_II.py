from dotenv import load_dotenv
load_dotenv();

import os

import src.features.build_features as FeaturesManager

from comet_ml import Experiment

if __name__ == "__main__":

    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data_df = FeaturesManager.build_features(seasons_year, season_type, with_player_stats=True, with_strength_stats=True)

    subset_df = features_data_df.query("`Game ID` == '2017021065'").reset_index(drop=True)

    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name="feature_engineering_data",
        workspace="ift6758-a22-g08"
    )
    
    experiment.log_dataframe_profile(
        subset_df, 
        name='wpg_v_wsh_2017021065',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )


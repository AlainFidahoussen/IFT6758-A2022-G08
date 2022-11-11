import src.data.NHLDataManager as DataManager
import numpy as np
import os
import pandas as pd

import warnings
warnings.filterwarnings('error')

from dotenv import load_dotenv
load_dotenv();


def build_features_one_season(season_year: int, season_type: str = "Regular", with_player_stats: bool = False) -> pd.DataFrame:
    """Build the features that will be used to train the models, for a specific season

    :param season_year: specific season year
    :type season_year: int
    :param season_type: 'Regular' or 'Playoffs'
    :type season_type: str
    :param with_player_stats: add player stats
    :type with_player_stats: bool
    :return: the features data frame
    :rtype: pd.DataFrame
    """

    data_manager = DataManager.NHLDataManager()

    dir_csv = os.path.join(data_manager.data_dir, "processed", "csv")
    filename = f'{season_year}_{season_type}_M2.csv'
    path_csv = os.path.join(dir_csv, filename)

    if os.path.exists(path_csv):
        features_data_df = pd.read_csv(path_csv, dtype={'Game ID': str})
    else:
        features_data_df = data_manager.get_season_dataframe(season_year=season_year, season_type=season_type)

        features_data_df.dropna(subset=['st_X', 'st_Y'], inplace=True)

        net_coordinates = np.array([89, 0])
        p2 = np.array([0, 0])

        # features_data_df['Shot Distance'] = np.linalg.norm(np.array([features_data_df['st_X'], features_data_df['st_Y']]) - net_coordinates, axis=1) # Goal is located at (89, 0)
        features_data_df['Shot distance'] = features_data_df.apply(lambda row: np.linalg.norm(np.array([row['st_X'], row['st_Y']]) - net_coordinates), axis=1)
        features_data_df['Shot angle'] = features_data_df.apply(lambda row: calculate_angle(np.array([row['st_X'], row['st_Y']]), net_coordinates, p2), axis=1)
        features_data_df['Is Goal'] = features_data_df.apply(lambda row: 1 if row['Type'] == 'GOAL' else 0, axis=1)
        # features_data_df['Is Goal'] = (features_data_df['Type'] == 'GOAL').astype(int)

        # features_data_df.drop(['Type'], axis=1, inplace=True) # I need this column for pivot_table

        features_data_df['Is Empty'] = features_data_df.apply(lambda row: 1 if row['Empty Net'] == True else 0, axis=1)
        # features_data_df['Is Empty'] = (features_data_df['Empty Net'] == True).astype(int)
        features_data_df.drop(['Empty Net'], axis=1, inplace=True) 

        features_data_df['Period seconds'] = pd.to_timedelta(features_data_df['Time'].apply(lambda x: f'00:{x}')).dt.seconds

        features_data_df['Game seconds'] = pd.to_timedelta(features_data_df['Time'].apply(lambda x: f'00:{x}')).dt.seconds
        features_data_df['Game seconds'] = features_data_df.apply(lambda row : row['Game seconds'] + 20*60*(row['Period']-1) if row['Period'] in [2, 3, 4] else (row['Game seconds'] if row['Period'] == 1 else row['Game seconds'] + 65*60), axis=1) # Bring time to the whole duration of the game 
        features_data_df.drop(['Time'], axis=1, inplace=True) 
        
        features_data_df['Last event angle'] = features_data_df.apply(lambda row: calculate_angle(np.array([row['Last event st_X'], row['Last event st_Y']]), net_coordinates, p2), axis=1)

        # features_data_df['Rebound'] = (features_data_df['Last event type'] == 'Shot').astype(int)
        features_data_df['Rebound'] = features_data_df.apply(lambda row: True if row['Last event type'] == 'Shot' else False, axis=1)
        
        features_data_df['Change in Shot Angle'] = features_data_df.apply(lambda row: np.abs(row['Shot angle'] - row['Last event angle']) if row['Rebound'] == True else 0, axis=1)

        features_data_df['Speed From Previous Event'] = features_data_df.apply(lambda row: calculate_speed(row['Last event distance'], row['Last event elapsed time']), axis=1)
        # features_data_df['Speed From Previous Event'] = features_data_df['Last event distance'] / features_data_df['Last event elapsed time']
        
        if with_player_stats:
            features_data_df = add_player_features(features_data_df, season_year)
            
        features_data_df.to_csv(path_csv, index=False)

    return features_data_df


def build_features(seasons_year: list[int], season_type: str = "Regular", with_player_stats: bool = True) -> pd.DataFrame:
    """Build the features that will be used to train the models, for several seasons

    :param seasons_year: list of specific season years
    :type seasons_year: list of int
    :param season_type: 'Regular' or 'Playoffs'
    :type season_type: str
    :param with_player_stats: add player stats
    :type with_player_stats: bool
    :return: the features data frame
    :rtype: pd.DataFrame
    """

    frames = [build_features_one_season(season_year, season_type, with_player_stats) for season_year in seasons_year]
    features_data_df = pd.concat(frames)
    features_data_df.reset_index(drop=True, inplace=True)
    return features_data_df


def calculate_angle(a, b, c):
    
    try:
        v0 = a - b
        v1 = c - b

        angle = np.degrees(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
        
    except:
        angle = np.nan
    
    return angle


def calculate_speed(a, b):
    
    try:
        result = a/b
    except: # either a division by zero (6982 rows with 'Last event elapsed time' == 0) or np.nan 
        result = np.nan
    
    return result


def add_player_features(features_data_df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """Add some additional features relative to the player that took the shot

    :param features_data_df: input dataframe
    :type features_data_df: features_data_df
    :param season_year: specific season year
    :type season_year: int
    :return: a data frame with the additional features
    :rtype: pd.DataFrame
    """

    data_manager = DataManager.NHLDataManager()

    player_goal_ratio_list = []
    goalie_goal_ratio_list = []

    for _, row in features_data_df.iterrows():

        shooter_id = int(row['Shooter ID'])
        player_stats = data_manager.load_player(shooter_id, season_year-1)


        try:
            shots = player_stats['stats'][0]['splits'][0]['stat']['shots']
            goals = player_stats['stats'][0]['splits'][0]['stat']['goals']
            if shots == 0:
                player_goal_ratio = np.nan
            else:
                player_goal_ratio = goals / shots
        except:
            player_goal_ratio = np.nan

        player_goal_ratio_list.append(player_goal_ratio)


        # Check if we have a goal keeper
        if row['Is Empty'] == 1:
            goalie_goal_ratio_list.append(np.nan)
            continue

        goalie_id = int(row['Goalie ID'])
        goalie_stats = data_manager.load_player(goalie_id, season_year-1)

        try:
            shots = goalie_stats['stats'][0]['splits'][0]['stat']['shotsAgainst']
            goals = goalie_stats['stats'][0]['splits'][0]['stat']['goalsAgainst']
            if shots == 0:
                goalie_goal_ratio = np.nan
            else:
                goalie_goal_ratio = goals / shots
        except:
            goalie_goal_ratio = np.nan

        goalie_goal_ratio_list.append(goalie_goal_ratio)
        

    features_data_add_df = features_data_df.copy()
    features_data_add_df['Shooter Goal Ratio Last Season'] = player_goal_ratio_list
    features_data_add_df['Goalie Goal Ratio Last Season'] = goalie_goal_ratio_list


    return features_data_add_df



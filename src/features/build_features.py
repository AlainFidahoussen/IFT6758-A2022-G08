import src.data.NHLDataManager as DataManager
import numpy as np
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv();


def build_features_one_season(season_year, season_type="Regular"):

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
        # features_data_df.drop(['Type'], axis=1, inplace=True) # I need this column for pivot_table

        features_data_df['Is Empty'] = features_data_df.apply(lambda row: 1 if row['Empty Net'] == True else 0, axis=1)
        features_data_df.drop(['Empty Net'], axis=1, inplace=True) 

        features_data_df['Game seconds'] = pd.to_timedelta(features_data_df['Time'].apply(lambda x: f'00:{x}')).dt.seconds
        features_data_df['Game seconds'] = features_data_df.apply(lambda row : row['Game seconds'] + 20*60*(row['Period']-1) if row['Period'] in [2, 3, 4] else (row['Game seconds'] if row['Period'] == 1 else row['Game seconds'] + 65*60), axis=1) # Bring time to the whole duration of the game 
        features_data_df.drop(['Time'], axis=1, inplace=True) 
        
        features_data_df['Last event angle'] = features_data_df.apply(lambda row: calculate_angle(np.array([row['Last event st_X'], row['Last event st_Y']]), net_coordinates, p2), axis=1)
        
        features_data_df['Rebound'] = features_data_df.apply(lambda row: True if row['Last event type'] == 'Shot' else False, axis=1)
        
        features_data_df['Change in Shot Angle'] = features_data_df.apply(lambda row: np.abs(row['Shot angle'] - row['Last event angle']) if row['Rebound'] == True else 0, axis=1)

        features_data_df['Speed From Previous Event'] = features_data_df.apply(lambda row: calculate_speed(row['Last event distance'], row['Last event elapsed time']), axis=1)
        

        features_data_df.to_csv(path_csv, index=False)

    return features_data_df


def build_features(seasons_year, season_type="Regular"):

    frames = [build_features_one_season(season_year, season_type) for season_year in seasons_year]
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

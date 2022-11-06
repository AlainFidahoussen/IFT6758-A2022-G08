import src.data.NHLDataManager as DataManager
import numpy as np
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv();

def build_features(seasons_year):

    data_manager = DataManager.NHLDataManager()

    dir_csv = os.path.join(data_manager.data_dir, "processed", "csv")
    filename = f'features_data.csv'
    path_csv = os.path.join(dir_csv, filename)
    if os.path.exists(path_csv):
        features_data_df = pd.read_csv(path_csv, dtype={'Game ID': str})
    else:
        frames = []
        season_type = "Regular"
        for season_year in seasons_year:
            data_season_df = data_manager.get_season_dataframe(season_year=season_year, season_type=season_type)
            frames.append(data_season_df)
        features_data_df = pd.concat(frames)


        features_data_df.dropna(subset=['st_X', 'st_Y'], inplace=True)
        features_data_df.reset_index(drop=True, inplace=True)

        net_coordinates = np.array([89, 0])
        p2 = np.array([0, 0])

        # features_data_df['Shot Distance'] = np.linalg.norm(np.array([features_data_df['st_X'], features_data_df['st_Y']]) - net_coordinates, axis=1) # Goal is located at (89, 0)
        features_data_df['Shot distance'] = features_data_df.apply(lambda row: np.linalg.norm(np.array([row['st_X'], row['st_Y']]) - net_coordinates), axis=1)
        features_data_df['Shot angle'] = features_data_df.apply(lambda row: calculate_angle(np.array([row['st_X'], row['st_Y']]), net_coordinates, p2), axis=1)
        features_data_df['Is Goal'] = features_data_df.apply(lambda row: 1 if row['Type'] == 'GOAL' else 0, axis=1)
        # features_data_df.drop(['Type'], axis=1, inplace=True) # I need this column for pivot_table

        features_data_df['Is Empty'] = features_data_df.apply(lambda row: 1 if row['Empty Net'] == True else 0, axis=1)
        features_data_df.drop(['Empty Net'], axis=1, inplace=True) 

        features_data_df['Game seconds'] = features_data_df.apply(lambda row: int(row['Time'].split(':')[0])*60 + int(row['Time'].split(':')[1]), axis=1)
        features_data_df.drop(['Time'], axis=1, inplace=True)

        features_data_df.to_csv(path_csv, index=False)

    return features_data_df


def calculate_angle(a, b, c):
    
    try:
        v0 = a - b
        v1 = c - b

        angle = np.degrees(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
        
    except:
        angle = np.nan
    
    return angle

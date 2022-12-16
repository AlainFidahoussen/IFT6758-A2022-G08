import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter 


def get_shots_coordinates(data_season_df:pd.DataFrame, team:str=None):

    # Return the mean for all the team
    if team is None:
        x_shots = data_season_df['st_X'].to_numpy().copy().astype(np.float32)
        y_shots = data_season_df['st_Y'].to_numpy().copy().astype(np.float32)

        x_nan = np.isnan(x_shots)
        y_nan = np.isnan(y_shots)

        x_shots = x_shots[(~x_nan) & (~y_nan)]
        y_shots = y_shots[(~x_nan) & (~y_nan)]

    # Return the mean just for the team
    else:
        data_one_team_df = data_season_df.loc[data_season_df['Team'] == team]
        if data_one_team_df.size == 0:
            return [[],[]]
        
        x_shots = data_one_team_df['st_X'].to_numpy().copy().astype(np.float32)
        y_shots = data_one_team_df['st_Y'].to_numpy().copy().astype(np.float32)

    # Remove the nan value for both x and y
    x_nan = np.isnan(x_shots)
    y_nan = np.isnan(y_shots)
    x_shots = x_shots[(~x_nan) & (~y_nan)]
    y_shots = y_shots[(~x_nan) & (~y_nan)]

    return [x_shots, y_shots]


def get_shots_hist2D(x_shots, y_shots, num_pts_x:int=40, num_pts_y:int=20):

    # We are only interested in shots in offensive zone, so we don't care about negative x coordinates
    x_min, x_max = 0.0, 100.
    y_min, y_max = -42.5, 42.5

    delta_x = (x_max-x_min) / num_pts_x
    delta_y = (y_max-y_min) / num_pts_y

    x_grid = np.arange(x_min-delta_x, x_max+delta_x, delta_x)
    y_grid = np.arange(y_min-delta_y, y_max+delta_y, delta_y)

    H, x_edge, y_edge = np.histogram2d(x_shots, y_shots, bins=[x_grid, y_grid])
    
    return H.T, x_edge[1:], y_edge[1:]


def compute_diff_shots(data_season_df:pd.DataFrame, num_pts_x:int=40, num_pts_y:int=20) -> dict:

    dict_diff = {}

    [x_shots_season, y_shots_season] = get_shots_coordinates(data_season_df)
    
    shots_hist2D_season, x_grid, y_grid = get_shots_hist2D(x_shots_season, y_shots_season, num_pts_x=num_pts_x, num_pts_y=num_pts_y)
    number_of_games_season = len(data_season_df['Game ID'].unique())
    shots_hist2D_season_by_hour = shots_hist2D_season / (number_of_games_season*2)

    teams = np.sort(data_season_df['Team'].unique())
    df_number_of_games_by_team = data_season_df[['Team', 'Game ID']].groupby('Team').describe()['Game ID']['unique']

    for team in teams:

        [x_shots_one_team, y_shots_one_team] = get_shots_coordinates(data_season_df, team)
    

        shots_hist2D_one_team, x_grid, y_grid = get_shots_hist2D(x_shots_one_team, y_shots_one_team, num_pts_x=num_pts_x, num_pts_y=num_pts_y)
        shots_hist2D_one_team_by_hour = shots_hist2D_one_team / df_number_of_games_by_team[team]

        diff = gaussian_filter(shots_hist2D_one_team_by_hour-shots_hist2D_season_by_hour, 1.)

        # Normalize between -1 and 1
        diff_min = diff.min()
        diff_max = diff.max()
        alpha = (-2./(diff_min-diff_max)) 
        beta = (diff_min + diff_max) / (diff_min - diff_max)
        diff_norm = alpha * diff + beta

        # Remove shots behind the goals
        mask = np.where(x_grid > 89)
        diff_norm[:, mask] = None
        dict_diff[team] = diff_norm
        

    return dict_diff, x_grid, y_grid



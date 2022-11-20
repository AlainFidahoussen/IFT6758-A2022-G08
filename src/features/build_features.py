import src.data.NHLDataManager as DataManager
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# import warnings
# warnings.filterwarnings('error')

from dotenv import load_dotenv
load_dotenv();


def build_features_one_season(season_year: int, season_type: str = "Regular", with_player_stats: bool = False, with_strength_stats: bool = False) -> pd.DataFrame:
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
        # features_data_df['Is Goal'] = features_data_df.apply(lambda row: 1 if row['Type'] == 'GOAL' else 0, axis=1)
        features_data_df['Is Goal'] = (features_data_df['Type'] == 'Goal').astype(int)

        # features_data_df.drop(['Type'], axis=1, inplace=True) # I need this column for pivot_table

        # features_data_df['Is Empty'] = features_data_df.apply(lambda row: 1 if row['Empty Net'] == True else 0, axis=1)
        features_data_df['Is Empty'] = (features_data_df['Empty Net'] == True).astype(int)
        features_data_df.drop(['Empty Net'], axis=1, inplace=True) 

        features_data_df['Period seconds'] = pd.to_timedelta(features_data_df['Time'].apply(lambda x: f'00:{x}')).dt.seconds

        features_data_df['Game seconds'] = pd.to_timedelta(features_data_df['Time'].apply(lambda x: f'00:{x}')).dt.seconds
        features_data_df['Game seconds'] = features_data_df.apply(lambda row : row['Game seconds'] + 20*60*(row['Period']-1) if row['Period'] in [2, 3, 4] else (row['Game seconds'] if row['Period'] == 1 else row['Game seconds'] + 65*60), axis=1) # Bring time to the whole duration of the game 
        features_data_df.drop(['Time'], axis=1, inplace=True) 
        
        features_data_df['Last event angle'] = features_data_df.apply(lambda row: calculate_angle(np.array([row['Last event st_X'], row['Last event st_Y']]), net_coordinates, p2), axis=1)

        features_data_df['Rebound'] = (features_data_df['Last event type'].str.contains('Shot')).astype(int)
        # features_data_df['Rebound'] = features_data_df.apply(lambda row: True if row['Last event type'] == 'Shot' else False, axis=1)
        
        features_data_df['Change in Shot Angle'] = features_data_df.apply(lambda row: np.abs(row['Shot angle'] - row['Last event angle']) if row['Rebound'] == True else 0, axis=1)

        features_data_df['Speed From Previous Event'] = features_data_df.apply(lambda row: calculate_speed(row['Last event distance'], row['Last event elapsed time']), axis=1)
        # features_data_df['Speed From Previous Event'] = features_data_df['Last event distance'] / features_data_df['Last event elapsed time']
        
        if with_player_stats:
            features_data_df = add_player_features(features_data_df, season_year)

        if with_strength_stats:
            features_data_df = add_stregth_features(features_data_df, season_year, season_type)
            
        features_data_df.to_csv(path_csv, index=False)

    return features_data_df


def build_features(seasons_year: list[int], season_type: str = "Regular", with_player_stats: bool = False, with_strength_stats: bool = False) -> pd.DataFrame:
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

    frames = [build_features_one_season(season_year, season_type, with_player_stats, with_strength_stats) for season_year in seasons_year]
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


def calculate_game_seconds(period: int, time: str) -> int:

    time_global = 0.

    time_seconds = pd.to_timedelta(f'00:{time}').seconds

    if period == 1:
        time_global = time_seconds
    elif period in [2, 3, 4]:
        time_global = time_seconds + 20*60*(period-1)
    else: # shootout : do we have that in the dataframe?
        time_global = time_seconds + 65*60

    return time_global


def calculate_speed(a, b):
    
    try:
        result = a/b
    except: # either a division by zero (6982 rows with 'Last event elapsed time' == 0) or np.nan 
        result = np.nan
    
    return result


def add_player_features(features_data_df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """Add some additional features relative to the player that took the shot and the goalie (goals/shots ratio) 

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


def add_stregth_features(features_data_df: pd.DataFrame, season_year: int, season_type: str) -> pd.DataFrame:

    data_manager = DataManager.NHLDataManager()

    features_data_add_lst = []
    columns_to_keep = list(features_data_df.columns) + ['Num players With', 'Num players Against', 'Elapsed time since Power Play']

    game_ids = features_data_df['Game ID'].unique()

    for game_id in game_ids:

        game_number = int(game_id[6:])

        df_goals = features_data_df.query(f"`Game ID` == '{game_id}' & Type == 'Goal'")
        df_shots = features_data_df.query(f"`Game ID` == '{game_id}' & Type == 'Shot'")
        df_penalties = data_manager.get_penalties_df(season_year, season_type, game_number)

        features_data_df['Num players With'] = np.nan
        features_data_df['Num players Against'] = np.nan

        if df_penalties is None:
            df = features_data_df.query(f"`Game ID` == '{game_id}'").copy()
            df['Num players With'] = 5
            df['Num players Against'] = 5
            df['Strength'] = 'Even'
            df['Elapsed time since Power Play'] = 0           
            features_data_add_lst.append(df)
            continue


        goals_shots_penalities_ordered_df = _get_goals_shots_penalities_ordered(df_goals, df_shots, df_penalties)
        goals_shots_penalities_ordered_df = _compute_strength_and_num_players(goals_shots_penalities_ordered_df)

        features_data_add_lst.append(goals_shots_penalities_ordered_df.query("Type == 'Shot' | Type == 'Goal'")[columns_to_keep])


    features_data_add_df = pd.concat(features_data_add_lst)
    features_data_add_df.reset_index(drop=True, inplace=True)

    return features_data_add_df


def _compute_strength_and_num_players(goals_shots_penalities_ordered_df: pd.DataFrame) -> pd.DataFrame:

    teams = goals_shots_penalities_ordered_df['Team'].unique()
    team0 = teams[0]
    team1 = teams[1]

    num_players_by_team = {team0: 5, team1: 5}
    start_penalty_time_by_team = {team0: 0, team1: 0}

    for count, row in goals_shots_penalities_ordered_df.iterrows():

        team_with = row['Team']
        if team_with == team0:
            team_against = team1
        else:
            team_against = team0

        # For simplicity, just ignore when a penalty happens in over-time
        if row['Period'] > 3:
            goals_shots_penalities_ordered_df.at[count, 'Num players With'] = 3
            goals_shots_penalities_ordered_df.at[count, 'Num players Against'] = 3
            goals_shots_penalities_ordered_df.at[count, 'Elapsed time since Power Play'] = 0
            goals_shots_penalities_ordered_df.at[count, 'Strength'] = 'Even'
            continue   

        if row['Type'] == 'PENALTY_Start':

            # The team just got a penalty. Record the start time
            if num_players_by_team[team_with] == 5:
                start_penalty_time_by_team[team_with] = row['Game seconds']

            # Ignore the penalty when there are just 3 players in the ice
            num_players_by_team[team_with] = max(num_players_by_team[team_with] - 1, 3)
            

        elif row['Type'] == 'PENALTY_End':
            # Don't exceed 5 players
            num_players_by_team[team_with] = min(num_players_by_team[team_with] + 1, 5)

            # The team just got back to 5 players, reset the start time
            if num_players_by_team[team_with] == 5:
                start_penalty_time_by_team[team_with] = 0

        if num_players_by_team[team_with] > num_players_by_team[team_against]:
            if row['Type'] != 'Goal': goals_shots_penalities_ordered_df.at[count, 'Strength'] = 'Power Play' # don't overwrite the Goal Strength, already available id the API
            goals_shots_penalities_ordered_df.at[count, 'Elapsed time since Power Play'] = row['Game seconds'] - start_penalty_time_by_team[team_against]

        elif num_players_by_team[team_with] < num_players_by_team[team_against]:
            if row['Type'] != 'Goal': goals_shots_penalities_ordered_df.at[count, 'Strength'] = 'Short Handed' # don't overwrite the Goal Strength, already available id the API
            goals_shots_penalities_ordered_df.at[count, 'Elapsed time since Power Play'] = 0
        else:
            if row['Type'] != 'Goal': goals_shots_penalities_ordered_df.at[count, 'Strength'] = 'Even' # don't overwrite the Goal Strength, already available id the API
            goals_shots_penalities_ordered_df.at[count, 'Elapsed time since Power Play'] = 0
        
        goals_shots_penalities_ordered_df.at[count, 'Num players With'] = num_players_by_team[team_with]
        goals_shots_penalities_ordered_df.at[count, 'Num players Against'] = num_players_by_team[team_against]

    return goals_shots_penalities_ordered_df


def _get_goals_shots_penalities_ordered(df_goals: pd.DataFrame, df_shots: pd.DataFrame, df_penalties: pd.DataFrame) -> pd.DataFrame:

    # Ignoe the Game Misconduct and Penalty Shot, as it does not cause any expulsion from the ice
    # df_penalties = df_penalties[(df_penalties['Severity'] != 'Misconduct') & (df_penalties['Severity'] != 'Penalty Shot')]
    df_penalties.drop(df_penalties.index[ (df_penalties['Severity'].str.contains('Misconduct')) | (df_penalties['Severity'].str.contains('Penalty Shot')) ], inplace=True)

    # # For a Match penalty, put the time to 5mn (was aked on Piazza)
    df_penalties['Minutes'] = df_penalties.apply(lambda row: 5 if row['Severity'] == 'Match' else row['Minutes'], axis=1)

    # df_penalties['Minutes'] = df_penalties.apply(lambda row: np.linalg.norm(np.array([row['st_X'], row['st_Y']]) - net_coordinates), axis=1)


    # Transform time to global
    df_penalties['Game seconds'] = df_penalties.apply(lambda row: calculate_game_seconds(row['Period'], row['Time']), axis=1)

    # Update the end time for minor penalty
    df_penalties_end = _compute_penalties_end(df_goals, df_penalties)

    df_penalties['Type'] = df_penalties['Type'] + '_Start'
    df_penalties_end['Type'] = df_penalties_end['Type'] + '_End'

    df_penalties = pd.concat([df_penalties, df_penalties_end]).sort_values(by='Game seconds').reset_index(drop=True)

    goals_shots_penalities_ordered_df = pd.concat([df_goals, df_shots, df_penalties]).sort_values(by='Game seconds').reset_index(drop=True)

    return goals_shots_penalities_ordered_df


def _compute_penalties_end(df_goals: pd.DataFrame, df_penalties: pd.DataFrame) -> pd.DataFrame:

    df_penalties_end = df_penalties.copy()

    # Compute the theoric end time of the penalty 
    df_penalties_end['Game seconds'] = df_penalties['Game seconds'] + 60 * df_penalties['Minutes']
    
    goals_time = df_goals['Game seconds'].to_numpy()
    goals_team = df_goals['Team'].to_numpy()

    for count, row in df_penalties_end.iterrows():

        if 'minor' in row['Severity'].lower():

            penalty_duration = 60 * row['Minutes']
            time_start = row['Game seconds']
            time_end = time_start + penalty_duration

            diff = time_end - goals_time

            diff_inds = np.where((diff > 0) & (diff < penalty_duration))[0]

            if penalty_duration == 120: # Just a regular minor
                if len(diff_inds) == 1: # If there is a goal within the 2 minutes 
                    if goals_team[diff_inds[0]] != row['Team']: # End the penalty only if the other team scored
                        df_penalties_end.at[count, 'Game seconds'] = goals_time[diff_inds[0]] + 1 # The new time for the end of the penalty if equal to the goal time (add one for sorting)


            elif penalty_duration == 240: # A double minor
                if len(diff_inds) == 1: # If there is only one goal with the 4 minutes, end
                    if goals_team[diff_inds[0]] != row['Team']: # End the penalty only if the other team scored
                        if (time_end - goals_time[diff_inds[0]]) <= 120: # The goal happened during the first penalty: end the second one
                            df_penalties_end.at[count, 'Game seconds'] = time_end - 120 # The new time for the end of the penalty if equal to the goal time (add one for sorting)
                        else: # The goal happened during the second penalty: end everything
                            df_penalties_end.at[count, 'Game seconds'] =  goals_time[diff_inds[0]] + 1
                elif len(diff_inds) == 2: # If there are two goals within the 4 minutes, end the double minor penalty
                    if (goals_team[diff_inds[0]] != row['Team']) & (goals_team[diff_inds[1]] != row['Team']):
                        df_penalties_end.at[count, 'Game seconds'] = goals_time[diff_inds[0]] + 1 # The new time for the end of the penalty if equal to the goal time (add one for sorting)
 
    return df_penalties_end


def GetFeaturesToKeep():

    return ['Period', 'Period seconds', 'st_X', 'st_Y', 'Shot Type', 'Shot distance', 'Shot angle', 'Is Empty',
            'Strength', 'Rebound', 'Speed From Previous Event', 'Change in Shot Angle', 
            'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season', 'Shooter Side', 'Shooter Ice Position', 
            'Num players With', 'Num players Against', 'Elapsed time since Power Play',
            'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 'Last event distance', 'Last event angle', 
            'Is Goal']


def GetCategoricalIndices() -> list:
    return [[1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]]

def GetNumericalIndices() -> list:
    return [0, 6, 7, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]


def GetTrainValid():
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    data_df = build_features(seasons_year, season_type, with_player_stats=True, with_strength_stats=True)

    features_to_keep = GetFeaturesToKeep()

    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)

    df_features = data_df[feature_names]
    df_targets = data_df[target_name]

    df_features = df_features.fillna(df_features.median())

    df_features['Shot Type'] = df_features['Shot Type'].fillna(df_features['Shot Type'].mode().iloc[0])
    dummy_shot_type = pd.get_dummies(df_features['Shot Type'], prefix='Shot_Type')
    df_features = df_features.merge(dummy_shot_type, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shot Type'])

    df_features['Strength'] = df_features['Strength'].fillna(df_features['Strength'].mode().iloc[0])
    dummy_strength = pd.get_dummies(df_features['Strength'], prefix='Strength')
    df_features = df_features.merge(dummy_strength, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Strength'])
    
    df_features['Shooter Side'] = df_features['Shooter Side'].fillna(df_features['Shooter Side'].mode().iloc[0])
    dummy_shooter_side = pd.get_dummies(df_features['Shooter Side'], prefix='Shooter_Side')
    df_features = df_features.merge(dummy_shooter_side, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shooter Side'])

    df_features['Shooter Ice Position'] = df_features['Shooter Ice Position'].fillna(df_features['Shooter Ice Position'].mode().iloc[0])
    dummy_shooter_ice_pos = pd.get_dummies(df_features['Shooter Ice Position'], prefix='Shooter_Ice_Position')
    df_features = df_features.merge(dummy_shooter_ice_pos, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shooter Ice Position'])

    # Update features_name
    feature_names = list(df_features.columns)
    feature_names = np.array(feature_names)

    X = df_features
    y = df_targets

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    return X_train, X_valid, y_train, y_valid


def GetTrainValid_II():
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    data_df = build_features(seasons_year, season_type, with_player_stats=True, with_strength_stats=True)

    features_to_keep = GetFeaturesToKeep()

    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)

    df_features = data_df[feature_names]
    df_targets = data_df[target_name]
    # ---------- split before to prevent data leakage
    X_train, X_valid, y_train, y_valid = train_test_split(df_features, df_targets, test_size=0.2, random_state=RANDOM_SEED, stratify=df_targets)
    # ---------- reset index after random shuffle split
    X_train.reset_index(drop=True, inplace=True)
    X_valid.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)
    # ---------- fill numerical col with X_train median
    X_train_median = X_train.median()
    X_train = X_train.fillna(X_train_median)
    X_valid = X_valid.fillna(X_train_median)
    # ---------- 
    shot_type_mode = X_train['Shot Type'].mode().iloc[0]
    X_train['Shot Type'] = X_train['Shot Type'].fillna(shot_type_mode)
    dummy_shot_type = pd.get_dummies(X_train['Shot Type'], prefix='Shot_Type')
    X_train = X_train.merge(dummy_shot_type, left_index=True, right_index=True)
    X_train = X_train.drop(columns=['Shot Type'])

    X_valid['Shot Type'] = X_valid['Shot Type'].fillna(shot_type_mode)
    dummy_shot_type = pd.get_dummies(X_valid['Shot Type'], prefix='Shot_Type')
    X_valid = X_valid.merge(dummy_shot_type, left_index=True, right_index=True)
    X_valid = X_valid.drop(columns=['Shot Type'])
    # ----------
    strength_mode = X_train['Strength'].mode().iloc[0]
    X_train['Strength'] = X_train['Strength'].fillna(strength_mode)
    dummy_strength = pd.get_dummies(X_train['Strength'], prefix='Strength')
    X_train = X_train.merge(dummy_strength, left_index=True, right_index=True)
    X_train = X_train.drop(columns=['Strength'])
    
    X_valid['Strength'] = X_valid['Strength'].fillna(strength_mode)
    dummy_strength = pd.get_dummies(X_valid['Strength'], prefix='Strength')
    X_valid = X_valid.merge(dummy_strength, left_index=True, right_index=True)
    X_valid = X_valid.drop(columns=['Strength'])
    # ----------
    shooter_side_mode = X_train['Shooter Side'].mode().iloc[0]
    X_train['Shooter Side'] = X_train['Shooter Side'].fillna(shooter_side_mode)
    dummy_shooter_side = pd.get_dummies(X_train['Shooter Side'], prefix='Shooter_Side')
    X_train = X_train.merge(dummy_shooter_side, left_index=True, right_index=True)
    X_train = X_train.drop(columns=['Shooter Side'])
    
    X_valid['Shooter Side'] = X_valid['Shooter Side'].fillna(shooter_side_mode)
    dummy_shooter_side = pd.get_dummies(X_valid['Shooter Side'], prefix='Shooter_Side')
    X_valid = X_valid.merge(dummy_shooter_side, left_index=True, right_index=True)
    X_valid = X_valid.drop(columns=['Shooter Side'])
    # ----------
    shooter_ice_pos_mode = X_train['Shooter Ice Position'].mode().iloc[0]
    X_train['Shooter Ice Position'] = X_train['Shooter Ice Position'].fillna(shooter_ice_pos_mode)
    dummy_shooter_ice_pos = pd.get_dummies(X_train['Shooter Ice Position'], prefix='Shooter_Ice_Position')
    X_train = X_train.merge(dummy_shooter_ice_pos, left_index=True, right_index=True)
    X_train = X_train.drop(columns=['Shooter Ice Position'])
    
    X_valid['Shooter Ice Position'] = X_valid['Shooter Ice Position'].fillna(shooter_ice_pos_mode)
    dummy_shooter_ice_pos = pd.get_dummies(X_valid['Shooter Ice Position'], prefix='Shooter_Ice_Position')
    X_valid = X_valid.merge(dummy_shooter_ice_pos, left_index=True, right_index=True)
    X_valid = X_valid.drop(columns=['Shooter Ice Position'])

    # Update features_name
    feature_names = list(X_train.columns)
    feature_names = np.array(X_train)

    return X_train, X_valid, y_train, y_valid


def GetTest(season_type = "Regular"):

    seasons_year = [2019]

    data_df = build_features(seasons_year, season_type, with_player_stats=True, with_strength_stats=True)

    features_to_keep = GetFeaturesToKeep()

    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)

    df_features = data_df[feature_names]
    df_targets = data_df[target_name]

    df_features = df_features.fillna(df_features.median())

    df_features['Shot Type'] = df_features['Shot Type'].fillna(df_features['Shot Type'].mode().iloc[0])
    dummy_shot_type = pd.get_dummies(df_features['Shot Type'], prefix='Shot_Type')
    df_features = df_features.merge(dummy_shot_type, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shot Type'])

    df_features['Strength'] = df_features['Strength'].fillna(df_features['Strength'].mode().iloc[0])
    dummy_strength = pd.get_dummies(df_features['Strength'], prefix='Strength')
    df_features = df_features.merge(dummy_strength, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Strength'])

    df_features['Shooter Side'] = df_features['Shooter Side'].fillna(df_features['Shooter Side'].mode().iloc[0])
    dummy_shooter_side = pd.get_dummies(df_features['Shooter Side'], prefix='Shooter_Side')
    df_features = df_features.merge(dummy_shooter_side, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shooter Side'])

    df_features['Shooter Ice Position'] = df_features['Shooter Ice Position'].fillna(df_features['Shooter Ice Position'].mode().iloc[0])
    dummy_shooter_ice_pos = pd.get_dummies(df_features['Shooter Ice Position'], prefix='Shooter_Ice_Position')
    df_features = df_features.merge(dummy_shooter_ice_pos, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shooter Ice Position'])


    # Update features_name
    feature_names = list(df_features.columns)
    feature_names = np.array(feature_names)

    X = df_features
    y = df_targets

    return X, y
   



if __name__ == '__main__':

    seasons_year = [2015, 2016, 2017, 2018, 2019]
    features_data_df = build_features(seasons_year, season_type = "Regular", with_player_stats=True, with_strength_stats=True)



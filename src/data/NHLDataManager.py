import datetime

import requests

import json
import os
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class NHLDataManager:

    def __init__(self, data_dir=""):
        """Constructor method.

        :param data_dir: directory where the data will be downloaded. Created if it does not exist
                         If not specified, the NHL_DATA_DIR environement variable will be used. 
                         If the NHL_DATA_DIR environement variable does not exist, it will be asked by the constructor
        :type data_dir: str (optional)
        """

        self.season_min = 1950
        self.season_max = datetime.date.today().year
        self._season_types = ["regular", "playoffs", "playoff"]

        if data_dir != "":
            self._data_dir = data_dir
            os.makedirs(self._data_dir, exist_ok=True)

        else:

            if 'NHL_DATA_DIR' in os.environ:
                self._data_dir = os.environ['NHL_DATA_DIR']
                #print(f'This is your NHL_DATA_DIR environment: {self._data_dir}')
                os.makedirs(self._data_dir, exist_ok=True)
            else:
                print('Please set the NHL_DATA_DIR environment variable')
                self._data_dir = ""


    @property
    def data_dir(self):
        return self._data_dir

    @property
    def season_types(self):
        return self._season_types

    @data_dir.setter
    def data_dir(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self._data_dir = data_dir


    def _get_url(self, game_id: str) -> str:
        """Returns the url used to get the data for a specifif game id

        :param game_id: should be built according to the specs:
            https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        :type game_id: str
        :return: url from where the data will be retrieved
        :rtype: str
        """
        return f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"
        

    def validate_season(self, season_year: int, season_type: str) -> bool:
        """Checks if the season is valide (4-digits and between min and max season)

        :param season_year: specific year in format XXXX
        :type season_year: int
        :return: True is season_year is valid, False otherwise
        :rtype: bool
        """
        if len(str(season_year)) != 4:
            print(f'Invalid season year, should have 4 digits: year={season_year}')
            return False

        if (season_year <= self.season_min) | (season_year > self.season_max):
            print(f'Invalid season year, should be between {self.season_min} and {self.season_max}: year={season_year}')
            return False

        if season_type.lower() not in self.season_types:
            print(f'Invalid season type, should be "Regular" or "Playoffs"')
            return False

        return True


    def get_game_numbers(self, season_year: int, season_type: str) -> list:
        """Returns the all game numbers played by each time, for a specific season

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param season_type: "Regular" or "Playoffs"
        :type season_type: str
        :return: game_numbers
        :rtype: list
        """

        # Pourrait probablement être déduit à partir des données l'API
        # nombre d'équipes * nombre de match / 2
        # 1271 = 31 * 82 / 2
        # 1230 = 30 * 82 / 2
        if type(season_year) is str:
            season_year = int(season_year)

        if not self.validate_season(season_year, season_type):
            return []

        if season_type == "Regular":
            number_of_games = 1271
            if season_year < 2017:
                number_of_games = 1230
            elif season_year < 2021:
                number_of_games = 1271
            else:
                number_of_games = 1312

            game_numbers = list(range(1, number_of_games + 1))
        else:
            game_numbers = []
            ronde = 4
            matchup = 8
            game = 7
            for i in range(1, ronde + 1):
                for j in range(1, int(matchup) + 1):
                    for k in range(1, game + 1):
                        code = int(f'{i}{j}{k}')
                        game_numbers.append(code)
                matchup /= 2

        return game_numbers


    def build_game_id(self, season_year: int, season_type: str, game_number: int) -> str:
        """Build the game_id, according to the specs:
        https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param season_type: "Regular" or "Playoffs"
        :type season_type: str
        :param game_number: specific game number
        :type game_number: int
        :return: game id, should be of length 8
        :rtype: str
        """

        if not self.validate_season(season_year, season_type):
            return ""

        if season_type == "Regular":
            return f'{season_year}02{str(game_number).zfill(4)}'
        else:
            return f'{season_year}03{str(game_number).zfill(4)}'


    def load_game(self, season_year: list, season_type: str, game_number: int) -> dict:
        """Download or read data of a specific game

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param season_type: "Regular" or "Playoffs"
        :type season_type: str
        :param game_number: specific game number
        :type game_number: int
        :return: a dictionary that contains the data (down)loaded
        :rtype: dict
        """

        if self.data_dir == "":
            print('The data directory is not defined, please defined it before to continue.')
            return {}

        if not self.validate_season(season_year, season_type):
            return {}

        path_output = os.path.join(self.data_dir, "raw", str(season_year), season_type)

        os.makedirs(path_output, exist_ok=True)
        game_id = self.build_game_id(season_year, season_type, game_number)
        game_id_path = os.path.join(path_output, f'{game_id}.json')

        # If the json has already been download, just read it 
        if os.path.exists(game_id_path):
            try:
                json_dict = json.load(open(game_id_path))
                return json_dict
            except json.JSONDecodeError: # if the json file is not valid, retrieve it from the API
                pass

        # If not, download and save the json
        url = self._get_url(f'{game_id}')
        r = requests.get(url)
        if r.status_code == 200:
            data_json = r.json()
            json.dump(data_json, open(game_id_path, "w"), indent=4)
            return data_json
        else:
            return {}



    def download_data(self, seasons_year: list, season_type: str) -> None:
        """Download all the data of season year and type
           If they are already downloaded, they will be skipped

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param season_type: "Regular" or "Playoffs"
        :type season_type: str
        :param path_output: specific game number
        :type path_output: str
        """

        if self.data_dir == "":
            print('The data directory is not defined, please defined it before to continue.')
            return None

        pbar_season = tqdm(seasons_year, position=0)
        for season_year in pbar_season:
            pbar_season.set_description(f'Season {season_year} - {season_type}')


            if not self.validate_season(season_year, season_type):
                continue

            path_data = os.path.join(self.data_dir, "raw", str(season_year), season_type)
            os.makedirs(path_data, exist_ok=True)

            if not self.validate_season(season_year, season_type):
                print(f'Cannot download season {season_year}')
                continue

            game_numbers = self.get_game_numbers(season_year, season_type)

            pbar_game = tqdm(game_numbers, position=1, leave=True)
            for game_number in pbar_game:
                pbar_game.set_description(f'Game {game_number}')

                # Build the game id and get the path to load/save the json file
                game_id = self.build_game_id(season_year, season_type, game_number)
                game_id_path = os.path.join(path_data, f'{game_id}.json')

                # If the json has not already been download yet, do it!
                if not os.path.exists(game_id_path):
                   url = self._get_url(f'{game_id}')
                   r = requests.get(url)
                   if r.status_code == 200:
                       data_json = r.json()
                       json.dump(data_json, open(game_id_path, "w"), indent=4)

        return None


    def load_data(self, season_year:int, season_type:str) -> dict:
        """ Load data of a whole season in a dictionary

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param season_type: "Regular" or "Playoffs"
        :type season_type: str
        :return: a dictionary that contains the data. The keys are the game number
        :rtype: dict
        """

        if not self.validate_season(season_year, season_type):
            return {}

        if self.data_dir == "":
            print('The data directory is not defined, please defined it before to continue.')
            return {}

        nhl_data = {}

        game_numbers = self.get_game_numbers(season_year, season_type)
        path_data = os.path.join(self.data_dir, "raw", str(season_year), season_type)
        os.makedirs(path_data, exist_ok=True)

        pbar_game = tqdm(game_numbers)
        for game_number in pbar_game:
            pbar_game.set_description(f'Game {game_number}')
            nhl_data[game_number] = self.load_game(season_year, season_type, game_number)

        return nhl_data


    def _get_game_ids(self, season_year:int, season_type:str) -> list:
        """Return the list of game_id for a specific season year and type (regular or playoffs)

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param path_data: path where the data will be [read if exist] - [stored if not]
        :type path_data: str
        :return: the list of game_id (8 digits)
        :rtype: list[int]
        """

        if not self.validate_season(season_year, season_type):
            return []

        if self.data_dir == "":
            print('The data directory is not defined, please defined it before to continue.')
            return []

        path_data = os.path.join(self.data_dir,  "raw", str(season_year), season_type)

        if not os.path.exists(path_data):
            self.download_data(seasons_year=[season_year], season_type=season_type)

        json_files = os.listdir(path_data)
        json_files = [f for f in json_files if f.endswith('.json')]

        games_id = [int(f[0:8]) for f in json_files]

        return games_id


    def get_game_numbers_from_data(self, season_year:int, season_type:str) -> list:
        """Return the list of game number for a specific season year and type (regular or playoffs)

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :return: the list of game number (3 or 4 digits)
        :rtype: list[int]
        """

        if not self.validate_season(season_year, season_type):
            return []

        if self.data_dir == "":
            print('The data directory is not defined, please defined it before to continue.')
            return []

        path_data = os.path.join(self.data_dir,  "raw", str(season_year), season_type)

        if not os.path.exists(path_data):
            self.download_data(seasons_year=[season_year], season_type=season_type)

        json_files = os.listdir(path_data)
        json_files = [f for f in json_files if f.endswith('.json')]

        games_number = [int(f[6:10]) for f in json_files]

        return games_number


    def get_teams_from_game(self, data_game : dict) -> dict:
        """Return the teams from the data

        :param data_game: data of a specific game already (down)loaded
        :type data_game: dict
        :return: a dictionary {abbr_home:name_home, abbr_away:name_away}
        :rtype: dict
        """
        try:
            team_name_away = data_game['gameData']['teams']['away']['name']
            team_abbr_away = data_game['gameData']['teams']['away']['abbreviation']

            team_name_home = data_game['gameData']['teams']['home']['name']
            team_abbr_home = data_game['gameData']['teams']['home']['abbreviation']

            return {team_abbr_home:team_name_home, team_abbr_away:team_name_away}
        except KeyError:
            return {}


    def get_teams(self, season_year:int, season_type:str, game_number:int) -> dict:

        """Return the teams of a specific game

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :param path_data: path where the data will be [read if exist] - [stored if not]
        :type path_data: str
        :return: a dictionary {abbr_home:name_home, abbr_away:name_away}
        :rtype: dict
        """

        data = self.load_game(season_year=season_year, season_type=season_type, game_number=game_number)
        return self.get_teams_from_game(data)



    def get_final_score_from_game(self, data_game : dict) -> dict:
        """Return the final score of a specific game

        :param data_game: data of a specific game already (down)loaded
        :type data_game: dict
        :return: a dictionary {abbr_home:score, abbr_away:score}
        :rtype: dict
        """

        try:
            team_abbr_away = data_game['gameData']['teams']['away']['abbreviation']
            team_abbr_home = data_game['gameData']['teams']['home']['abbreviation']

        except KeyError:
            return {}


        try:
            score_away = data_game['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats']['goals']
            score_home = data_game['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats']['goals']


        except KeyError:
            score_away = 0
            score_home = 0

        return {team_abbr_home:score_home, team_abbr_away:score_away}


    def get_final_score(self, season_year:int, season_type:str, game_number:int) -> dict:
        """Return the final score of a specific game as a dictionary

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :return: a dictionary {abbr_home:score, abbr_away:score}
        :rtype: dict
        """

        data = self.load_game(season_year=season_year, season_type=season_type, game_number=game_number)
        return self.get_final_score_from_game(data)


    def get_goals_and_shots(self, season_year:int, season_type:str, game_number:int) -> tuple:
        """Return the goals and shots event from a specific game

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :return: a tuple {goals events, shots events}
        :rtype: tuple
        """

        data = self.load_game(season_year, season_type, game_number)

        try:
            data = data['liveData']['plays']
            num_events = len(data['allPlays'])

            list_goals = data['scoringPlays']
            goal_events = [data['allPlays'][g] for g in list_goals]
            shot_events = [data['allPlays'][ev] for ev in range(num_events) if data['allPlays'][ev]['result']['event'] == 'Shot']
        except KeyError:
            return ([], [])

        return (goal_events, shot_events)


    def get_goals_and_shots_df(self, season_year:int, season_type:str, game_number:int) -> pd.DataFrame:
        """Return the goals and shots event from a specific game

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :return: a data frame
        :rtype: pd.DataFrame
        """

        (goal_events, shot_events) = self.get_goals_and_shots(season_year, season_type, game_number)

        if (len(goal_events) == 0) & (len(shot_events) == 0):
            return None

        game_id = self.build_game_id(season_year, season_type, game_number)

        num_events = len(goal_events) + len(shot_events)
        df = pd.DataFrame(index=range(num_events),
                          columns=['Game ID', 'Event Index', 'Time', 'Period', 'Team', 'Type', 'Shot Type', 'Shooter', 'Goalie',
                                   'Empty Net', 'Strength', 'X', 'Y'])

        count = 0
        for goal in goal_events:
            # Difference between eventId and eventIdx
            df.loc[count]['Event Index'] = goal['about']['eventIdx']
            df.loc[count]['Time'] = goal['about']['periodTime']
            df.loc[count]['Period'] = goal['about']['period']
            df.loc[count]['Game ID'] = game_id
            df.loc[count]['Team'] = f"{goal['team']['name']} ({goal['team']['triCode']})"

            try:
                df.loc[count]['X'] = goal['coordinates']['x']
                df.loc[count]['Y'] = goal['coordinates']['y']
            except KeyError:
                pass

            df.loc[count]['Type'] = 'GOAL'
            df.loc[count]['Shooter'] = goal['players'][0]['player']['fullName']
            df.loc[count]['Goalie'] = goal['players'][-1]['player']['fullName']
            if 'emptyNet' in goal['result']:
                df.loc[count]['Empty Net'] = goal['result']['emptyNet']
            else:
                df.loc[count]['Empty Net'] = True

            if 'secondaryType' in goal['result']:
                df.loc[count]['Shot Type'] = goal['result']['secondaryType']

            df.loc[count]['Strength'] = goal['result']['strength']['name']

            count += 1

        for shot in shot_events:
            df.loc[count]['Event Index'] = shot['about']['eventIdx']
            df.loc[count]['Time'] = shot['about']['periodTime']
            df.loc[count]['Period'] = shot['about']['period']
            df.loc[count]['Game ID'] = game_id
            df.loc[count]['Team'] = f"{shot['team']['name']} ({shot['team']['triCode']})"

            try:
                df.loc[count]['X'] = shot['coordinates']['x']
                df.loc[count]['Y'] = shot['coordinates']['y']
            except KeyError:
                pass

            df.loc[count]['Type'] = 'SHOT'
            df.loc[count]['Shooter'] = shot['players'][0]['player']['fullName']
            df.loc[count]['Goalie'] = shot['players'][-1]['player']['fullName']

            if 'secondaryType' in goal['result']:
                df.loc[count]['Shot Type'] = goal['result']['secondaryType']

            count += 1

        return df
        

    def get_goals_and_shots_df_standardised(self, season_year:int, season_type:str, game_number:int) -> pd.DataFrame:
        """Return the same dataframe as get_goals_and_shots_df, but with shot coordinates standardised (goal is always on the right side of the rink)
        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :return: a data frame
        :rtype: pd.DataFrame
        """

        # Loading data
        game_data = self.load_game(season_year, season_type, game_number)
        if len(game_data) == 0:
            return None

        goals_and_shots = self.get_goals_and_shots_df(season_year, season_type, game_number)
        if goals_and_shots is None:
            return None

        # Get period and team info from game data
        try:
            periods = game_data['liveData']['linescore']['periods']
            home_sides = [period['home']['rinkSide'] == 'left' for period in periods] # True for left, False for right
            away_sides = [period['away']['rinkSide'] == 'left' for period in periods]
            # check if there's a shootout period, if yes, same side as 3rd period
            if 'startTime' in game_data['liveData']['linescore']['shootoutInfo']:
                home_side_shootout = game_data['liveData']['linescore']['periods'][2]['home']['rinkSide'] == 'left'
                away_side_shootout = game_data['liveData']['linescore']['periods'][2]['away']['rinkSide'] == 'left'
                home_sides.append(home_side_shootout)
                away_sides.append(away_side_shootout)
            else:
                pass

            home_team = game_data['gameData']['teams']['home']['triCode'] # Tricode (e.g. MTL)
            away_team = game_data['gameData']['teams']['away']['triCode']

            # Computed "standardised" coordinates
            period_indices = goals_and_shots['Period'] - 1
            is_home = goals_and_shots['Team'].str.contains(f"({home_team})")
            sides = np.where(is_home, np.take(home_sides, period_indices), np.take(away_sides, period_indices))
             # boolean array: True if team is on left
            multiplier = (sides - 0.5) * 2
            goals_and_shots['st_X'] = multiplier * goals_and_shots['X']
            goals_and_shots['st_Y'] = multiplier * goals_and_shots['Y']
        except: # if no rinkSide info
            goals_and_shots['st_X'] = np.nan
            goals_and_shots['st_Y'] = np.nan

        return goals_and_shots


    def get_season_dataframe(self, season_year:int, season_type:str) -> pd.DataFrame:
        """Return the same dataframe as get_goals_and_shots_df_standardised, but for the whole season
           The dataframe is also saved as a CSV file in self.data_dir/processed
        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :return: a data frame
        :rtype: pd.DataFrame
        """

        if not self.validate_season(season_year, season_type):
            return []

        if self.data_dir == "":
            print('The data directory is not defined, please defined it before to continue.')
            return []

        dir_csv = os.path.join(self.data_dir, "processed", "csv")
        filename = f'{season_year}_{season_type}.csv'
        path_csv = os.path.join(dir_csv, filename)
        if os.path.exists(path_csv):
            data_season_df = pd.read_csv(path_csv, index_col=0, dtype={'Game ID': str})

        else:
            os.makedirs(dir_csv, exist_ok=True)

            game_numbers = self.get_game_numbers(season_year=season_year, season_type=season_type)
            data_season_df = self.get_goals_and_shots_df_standardised(season_year=season_year, season_type=season_type, game_number=game_numbers[0])

            pbar_game = tqdm(game_numbers, position=0, leave=True)
            pbar_game.set_description(f'Game {game_numbers[0]}')

            for game_number in tqdm(game_numbers[1:]):
                pbar_game.set_description(f'Game {game_number}')

                temp_df = self.get_goals_and_shots_df_standardised(season_year=season_year, season_type=season_type, game_number=game_number)
                if temp_df is None:
                    continue
                data_season_df = pd.concat([data_season_df, temp_df], ignore_index=True)

            data_season_df.to_csv(path_csv)

        return data_season_df



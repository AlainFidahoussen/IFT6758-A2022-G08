import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import datetime

import requests

import json
import os

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class NHLDataManager:

    def __init__(self):
        """Constructor method.
        """

        self.season_min = 1950
        self.season_max = datetime.date.today().year
        self._season_types = ["Regular", "Playoffs"]

    @property
    def season_types(self):
        return self._season_types


    def _get_game_url(self, game_id: str) -> str:
        """Returns the url used to get the data for a specifif game id

        :param game_id: should be built according to the specs:
            https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        :type game_id: str
        :return: url from where the data will be retrieved
        :rtype: str
        """
        return f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"


    
    def _get_player_stats_url(self, player_id: int, season_year: int) -> str:
        """Returns the url used to get the data for a specific player id and a specific season

        :param player_id: player id
        :type player_id: int
        :param season_year: season year
        :type season_year: int
        :return: url from where the data will be retrieved
        :rtype: str
        """
        return f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats/?stats=statsSingleSeason&season={season_year}{season_year+1}"


        
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

        if season_type not in self.season_types:
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


    def get_game_id(self, season_year: int, season_type: str, game_number: int) -> str:
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


    def load_player(self, player_id: int, season_year: int) -> dict:
        """Download or read stats data of a specific player

        :param player_id: player id
        :type player_id: int
        :param season_year: season year
        :type season_year: int
        :return: a dictionary that contains the data (down)loaded
        :rtype: dict
        """

        # If not, download and save the json
        url = self._get_player_stats_url(player_id, season_year)
        r = requests.get(url)
        if r.status_code == 200:
            data_json = r.json()
            return data_json
        else:
            return {}


    def load_game(self, season_year: int, season_type: str, game_number: int) -> dict:
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

        if not self.validate_season(season_year, season_type):
            print('Invalid season.')
            return {}

        game_id = self.get_game_id(season_year, season_type, game_number)

        # If not, download and save the json
        url = self._get_game_url(f'{game_id}')
        r = requests.get(url)
        if r.status_code == 200:
            data_json = r.json()
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


        pbar_season = tqdm(seasons_year, position=0)
        for season_year in pbar_season:
            pbar_season.set_description(f'Season {season_year} - {season_type}')


            if not self.validate_season(season_year, season_type):
                continue

            if not self.validate_season(season_year, season_type):
                print(f'Cannot download season {season_year}')
                continue

            game_numbers = self.get_game_numbers(season_year, season_type)

            pbar_game = tqdm(game_numbers, position=1, leave=True)
            for game_number in pbar_game:
                pbar_game.set_description(f'Game {game_number}')

                # Build the game id and get the path to load/save the json file
                game_id = self.get_game_id(season_year, season_type, game_number)

                # If the json has not already been download yet, do it!
                url = self._get_game_url(f'{game_id}')
                r = requests.get(url)
                if r.status_code == 200:
                    data_json = r.json()
                    with open(game_id_path, "w") as f:
                        json.dump(data_json, f, indent=4)

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

        nhl_data = {}

        game_numbers = self.get_game_numbers(season_year, season_type)

        pbar_game = tqdm(game_numbers)
        for game_number in pbar_game:
            pbar_game.set_description(f'Game {game_number}')
            nhl_data[game_number] = self.load_game(season_year, season_type, game_number)

        return nhl_data



    def get_teams(self, season_year:int, season_type:str, game_number:int) -> list:
        """Return the teams from from a specific game

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :return: a list [home, away]
        :rtype: list
        """
        try:
            data_game = self.load_game(season_year, season_type, game_number)

            team_name_away = data_game['gameData']['teams']['away']['name']
            team_abbr_away = data_game['gameData']['teams']['away']['abbreviation']
            team_away = f'{team_name_away} ({team_abbr_away})'

            team_name_home = data_game['gameData']['teams']['home']['name']
            team_abbr_home = data_game['gameData']['teams']['home']['abbreviation']
            team_home = f'{team_name_home} ({team_abbr_home})'

            return [team_home, team_away]
        except KeyError:
            return []


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
            players = data['gameData']['players']

            plays_data = data['liveData']['plays']
            num_events = len(plays_data['allPlays'])

            list_goals = plays_data['scoringPlays']
            goal_events = [plays_data['allPlays'][g] for g in list_goals]
            shot_events = [plays_data['allPlays'][ev] for ev in range(num_events) if plays_data['allPlays'][ev]['result']['event'] == 'Shot']
            all_events = [plays_data['allPlays'][ev] for ev in range(num_events)]
        except KeyError:
            return ([], [], [], [])

        return (goal_events, shot_events, all_events, players)


    def get_penalties(self, season_year:int, season_type:str, game_number:int) -> tuple:
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
            list_events = data['liveData']['plays']['penaltyPlays']
            penalties_events = [data['liveData']['plays']['allPlays'][ev] for ev in list_events]
        except KeyError:
            return []

        return penalties_events



    def get_penalties_df(self, season_year:int, season_type:str, game_number:int) -> pd.DataFrame:
        """Return the penalties event from a specific game

        :param season_year: specific season year
        :type season_year: int
        :param season_type: 'Regular' or 'Playoffs'
        :type season_type: str
        :param game_number: specific game number (could be get from the get_game_numbers() function)
        :type game_number: int
        :return: a data frame
        :rtype: pd.DataFrame
        """

        penalty_events = self.get_penalties(season_year, season_type, game_number)

        num_penalties = len(penalty_events)

        if num_penalties == 0:
            return None

        df = pd.DataFrame(index=range(num_penalties), columns=['Game ID', 'Event Index', 'Type', 'Player', 'Team', 'Period', 'Time', 'Global Time', 'Severity', 'Minutes'])

        game_id = self.get_game_id(season_year, season_type, game_number)

        for count, penalty in enumerate(penalty_events):

            player = penalty['players'][0]['player']['fullName']

            team_name = penalty['team']['name']
            team_abb = penalty['team']['triCode']

            period = penalty['about']['period']
            time = penalty['about']['periodTime']

            severity = penalty['result']['penaltySeverity']
            minutes = penalty['result']['penaltyMinutes']

            df.loc[count]['Game ID'] = game_id
            df.loc[count]['Event Index'] = penalty['about']['eventIdx']
            df.loc[count]['Type'] = 'PENALTY'
            df.loc[count]['Player'] = player
            df.loc[count]['Team'] = f'{team_name} ({team_abb})'
            df.loc[count]['Period'] = period
            df.loc[count]['Time'] = time
            df.loc[count]['Severity'] = severity
            df.loc[count]['Minutes'] = minutes

            # df.loc[count]['Global Time'] = FeaturesManager.calculate_global_time(period, time)

        # Sometimes, the same player could have, at the same time, several penalties of different severity

        # Just keep the more severe one (longest minutes)
        # cf. 2018, "Regular", 2
        Minutes_Max = df.groupby(['Player', 'Time']).Minutes.transform(max)
        df = df.loc[df.Minutes == Minutes_Max].reset_index(drop=True)

        # Or it should be the most recent event?
        # Event_Max = df.groupby(['Player', 'Time'])['Event Index'].transform(max)
        # df = df.loc[df['Event Index'] == Event_Max].reset_index(drop=True)


        return df



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

        (goal_events, shot_events, all_events, players) = self.get_goals_and_shots(season_year, season_type, game_number)


        if (len(goal_events) == 0) & (len(shot_events) == 0):
            return None

        if (len(all_events) == 0):
            return None

        # Transform the events in dictionary
        all_events_dict = {a['about']['eventIdx']: a for a in all_events}

        game_id = self.get_game_id(season_year, season_type, game_number)

        goals_shots_events = goal_events + shot_events
        num_events = len(goals_shots_events)
        df = pd.DataFrame(index=range(num_events),
                          columns=['Game ID', 'Event Index', 'Time', 'Period', 'Team', 'Type', 'Shot Type', 'Shooter', 'Shooter ID', 'Shooter Side', 'Shooter Ice Position', 'Goalie', 'Goalie ID', 
                                   'Empty Net', 'Strength', 'X', 'Y', 'Last event type', 'Last event X', 'Last event Y', 'Last event elapsed time', 'Last event distance'])


        count = 0
        for event in goals_shots_events:

            # Difference between eventId and eventIdx
            event_idx = event['about']['eventIdx']
            df.loc[count]['Event Index'] = event_idx

            df.loc[count]['Time'] = event['about']['periodTime']
            df.loc[count]['Period'] = event['about']['period']
            df.loc[count]['Game ID'] = game_id
            df.loc[count]['Team'] = f"{event['team']['name']} ({event['team']['triCode']})"

            df.loc[count]['Type'] = event['result']['event']
            df.loc[count]['Shooter'] = event['players'][0]['player']['fullName']
            df.loc[count]['Shooter ID'] = event['players'][0]['player']['id']

            df.loc[count]['Goalie'] = event['players'][-1]['player']['fullName']
            df.loc[count]['Goalie ID'] = event['players'][-1]['player']['id']

            if 'emptyNet' in event['result']:
                df.loc[count]['Empty Net'] = event['result']['emptyNet']
            else:
                df.loc[count]['Empty Net'] = False

            if 'secondaryType' in event['result']:
                if event['result']['secondaryType'] == 'Poke':
                    df.loc[count]['Shot Type'] = 'Wrist Shot'
                else:
                    df.loc[count]['Shot Type'] = event['result']['secondaryType']

            # Strength exists only for Goals
            if df.loc[count]['Type'] == 'Goal':
                df.loc[count]['Strength'] = event['result']['strength']['name']

            # For simplicity, ignore the stoppage events
            last_event = all_events_dict[event_idx-1]
            last_event_type = last_event['result']['event']
            if last_event_type.lower() == 'stoppage':
                last_event = all_events_dict[event_idx-2]
                last_event_type = last_event['result']['event']
                    

            df.loc[count]['Last event type'] = last_event['result']['event']

            try:
                df.loc[count]['X'] = event['coordinates']['x']
                df.loc[count]['Y'] = event['coordinates']['y']

            except KeyError:
                pass

            try:
                if last_event_type.lower() == 'period start':
                    df.loc[count]['Last event X'] = 0
                    df.loc[count]['Last event Y'] = 0
                else:
                    df.loc[count]['Last event X'] = last_event['coordinates']['x']
                    df.loc[count]['Last event Y'] = last_event['coordinates']['y']

                df.loc[count]['Last event distance'] = np.sqrt( (df.loc[count]['X']-df.loc[count]['Last event X'])**2 + 
                                                                (df.loc[count]['Y']-df.loc[count]['Last event Y'])**2 )
            except KeyError:
                pass


            time_event_s = float(event['about']['periodTime'].split(':')[0]) * 60 + int(event['about']['periodTime'].split(':')[1])
            time_last_event_s = float(last_event['about']['periodTime'].split(':')[0]) * 60 + int(last_event['about']['periodTime'].split(':')[1])
            df.loc[count]['Last event elapsed time'] = time_event_s - time_last_event_s + 0.5

            try:
                shooter_id = f"ID{df.loc[count]['Shooter ID']}"
                shooter_side = players[shooter_id]['shootsCatches']
                df.loc[count]['Shooter Side'] = shooter_side

                shooter_position = players[shooter_id]['primaryPosition']['code']
                df.loc[count]['Shooter Ice Position'] = shooter_position

            except KeyError:
                pass

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

        try:
            # Get period and team info from game data
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

            home_team_abb = game_data['gameData']['teams']['home']['triCode'] # Tricode (e.g. MTL)
            away_team_abb = game_data['gameData']['teams']['away']['triCode']

            # Computed "standardised" coordinates
            period_indices = goals_and_shots['Period'] - 1
            is_home = goals_and_shots['Team'].str.contains(home_team_abb)

            sides = np.where(is_home, np.take(home_sides, period_indices), np.take(away_sides, period_indices))

                # boolean array: True if team is on left
            multiplier = (sides - 0.5) * 2

            goals_and_shots['st_X'] = multiplier * goals_and_shots['X']
            goals_and_shots['st_Y'] = multiplier * goals_and_shots['Y']
            
            goals_and_shots['Last event st_X'] = multiplier * goals_and_shots['Last event X']
            goals_and_shots['Last event st_Y'] = multiplier * goals_and_shots['Last event Y']

        except:
            goals_and_shots['st_X'] = np.nan
            goals_and_shots['st_Y'] = np.nan
            
            goals_and_shots['Last event st_X'] = np.nan
            goals_and_shots['Last event st_Y'] = np.nan

        return goals_and_shots


    def get_season_dataframe(self, season_year:int, season_type:str) -> pd.DataFrame:
        """Return the same dataframe like get_goals_and_shots_df_standardised, but for the whole season
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


        game_numbers = self.get_game_numbers(season_year=season_year, season_type=season_type)

        data_season_list = [self.get_goals_and_shots_df_standardised(season_year=season_year, season_type=season_type, game_number=game_number) 
                            for game_number in game_numbers]

        data_season_df = pd.concat([d for d in data_season_list if d is not None], ignore_index=True)

        return data_season_df



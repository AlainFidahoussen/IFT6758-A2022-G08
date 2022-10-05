import datetime
import platform
import time

import requests

import json
import os
from tqdm.auto import tqdm

class NHLDataManager:

    def __init__(self):
        self.nhl_data = {}
        self.season_min = 1950
        self.season_max = datetime.date.today().year

        env = input('Enter NHL_DATA_DIR environment: ')

        if env == '':
            print('NHL_DATA_DIR environment is not defined ...')
            if "linux" in platform.platform().lower():
                os.environ['NHL_DATA_DIR'] = "/tmp/nhl_data_dir"
                os.makedirs(os.environ['NHL_DATA_DIR'], exist_ok=True)
                print(f"And had been set by default to {os.environ['NHL_DATA_DIR']}")
            elif "windows" in platform.platform().lower():
                os.environ['NHL_DATA_DIR'] = "C:/Temp/nhl_data_dir"
                os.makedirs(os.environ['NHL_DATA_DIR'], exist_ok=True)
                print(f"And had been set by default to {os.environ['NHL_DATA_DIR']}")
            else:
                print('Please defined it before to continue.')

        else:
            os.environ['NHL_DATA_DIR'] = env


    def get_data(self):
        return self.nhl_data


    def get_url(self, game_id: str) -> str:
        """ Returns the url used to get the data
                Input:
                  - game_id (str): should be built according to the specs: 
                    https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
                    
                Output:
                  - url (str) from where the data will be retrieved
        """
        return f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"
        

    def validate_season_year(self, season_year: int) -> bool:
        """ Checks if the season is valide (4-digits and between min and max season)
            Input:
              - season_year (int): specific year in format XXXX
              
            Output:
              - True is season_year is valid, False otherwise
        """
        
        if len(str(season_year)) != 4:
            print(f'Invalid season year, should have 4 digits: year={season_year}')
            return False

        if (season_year <= self.season_min) | (season_year > self.season_max):
            print(f'Invalid season year, should be between {self.season_min} and {self.season_max}: year={season_year}')
            return False

        return True

    def get_number_of_games(self, season_year: int, is_regular: bool) -> list:
        """ Returns the all game numbers played by each time, for a specific season
            Input:
              - season_year (int): specific year in format XXXX

            Output:
              - game_numbers (list)
        """
        
        # Pourrait probablement être déduit à partir des données l'API
        # nombre d'équipes * nombre de match / 2
        # 1271 = 31 * 82 / 2
        # 1230 = 30 * 82 / 2
        if type(season_year) is str:
            season_year = int(season_year)

        if is_regular:
            number_of_games = 1271
            game_numbers = [i for i in range(1, number_of_games + 1)]
            if season_year < 2017:
                number_of_games = 1230
                game_numbers = [i for i in range(1, number_of_games + 1)]
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


    def build_game_id(self, season_year: int, is_regular: bool, game_number: int) -> str:
        """ Build the game_id, according to the specs: 
            https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
            Input:
              - season_year (int): specific year in format XXXX
              - is_regular (bool): True if regular season, False if pre-season
              - game_number (int): game number (should be between 1 and self.get_number_of_games() 
                                   for a regular season)
            Output:
              - game_id (str) - should be of length 8
        """

        if is_regular:
            return f'{season_year}02{str(game_number).zfill(4)}'
        else:
            return f'{season_year}03{str(game_number).zfill(4)}'


    def download_data(self, seasons_year: list, is_regular: bool, path_output="") -> bool:
        """ Download data of season "season_year/season_year+1" and put them in the 
            directory specified path_output.
            By default, this is directory will be specified by the environment variable NHL_DATA_DIR.
            If they are already available, they will just be read. 
            
            Input:
                - seasons_year (list): list of seasons to downlaod (in format XXXX)
                - is_regular (bool): True if regular season, False if pre-season
                - path_output (str): directory that will contains the data downloaded
                                     will be created if it does not exist
            Output:
                - True if everything worked, False otherwise
        """
        if path_output == "":
            if "NHL_DATA_DIR" in os.environ:
                path_output = os.environ["NHL_DATA_DIR"]
            else:
                print('NHL_DATA_DIR environment is not defined, please defined it before to continue.')
                return False



        for season_year in tqdm(seasons_year, desc='Season', position=0):

            if is_regular:
                path_output_season = os.path.join(path_output, str(season_year), "regular")
            else:
                path_output_season = os.path.join(path_output, str(season_year), "playoffs")

            os.makedirs(path_output_season, exist_ok=True)


            if not self.validate_season_year(season_year):
                print(f'Cannot download season {season_year}')
                continue

            self.nhl_data[season_year] = []
            game_numbers = self.get_number_of_games(season_year, is_regular)

            for game_number in tqdm(game_numbers, desc=f'Season {season_year} - Game', position=1, leave=True):

                # Build the game id and get the path to load/save the json file
                game_id = self.build_game_id(season_year, is_regular, game_number)
                game_id_path = os.path.join(path_output_season, f'{game_id}.json')

                # If the json has already been download, just read it and go the next one
                if os.path.exists(game_id_path):
                    self.nhl_data[season_year].append(json.load(open(game_id_path)))
                    continue

                # If not, download and save the json
                url = self.get_url(f'{game_id}')
                r = requests.get(url)
                if r.status_code == 200:
                    data_json = r.json()
                    json.dump(data_json, open(game_id_path, "w"), indent=4)
                    self.nhl_data[season_year].append(data_json)

        return True



    def __add__(self, other_data):
        for keys in other_data.nhl_data.keys():
            if keys not in self.nhl_data.keys():
                self.nhl_data[keys] = other_data.nhl_data[keys]

        return self
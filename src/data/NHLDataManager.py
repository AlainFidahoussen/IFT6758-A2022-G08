import datetime
import platform
import time

import requests

import json
import os
from tqdm.auto import tqdm

class NHLDataManager:

    def __init__(self):
        """Constructor method. Asks for the NHL_DATA_DIR environment variable if it does not exist yet.
        """

        self._nhl_data = {}
        self.season_min = 1950
        self.season_max = datetime.date.today().year

        if 'NHL_DATA_DIR' in os.environ:
            env = os.environ['NHL_DATA_DIR']
            #print(f'This is your NHL_DATA_DIR environment: {env}')
            pass
        else:
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


    @property
    def nhl_data(self):
        return self._nhl_data


    def get_url(self, game_id: str) -> str:
        """Returns the url used to get the data for a specifif game id

        :param game_id: should be built according to the specs:
            https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        :type game_id: str
        :return: url from where the data will be retrieved
        :rtype: str
        """
        return f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"
        

    def validate_season_year(self, season_year: int) -> bool:
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

        return True

    def get_number_of_games(self, season_year: int, is_regular: bool) -> list:
        """Returns the all game numbers played by each time, for a specific season

        :param season_year: specific year in format XXXX
        :type season_year: int
        :return: game_numbers
        :rtype: list
        """

        # Pourrait probablement être déduit à partir des données l'API
        # nombre d'équipes * nombre de match / 2
        # 1271 = 31 * 82 / 2
        # 1230 = 30 * 82 / 2
        if type(season_year) is str:
            season_year = int(season_year)

        if is_regular:
            number_of_games = 1271
            if season_year < 2017:
                number_of_games = 1230

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


    def build_game_id(self, season_year: int, is_regular: bool, game_number: int) -> str:
        """Build the game_id, according to the specs:
        https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param is_regular: True if regular season, False if playoffs
        :type is_regular: bool
        :param game_number: specific game number
        :type game_number: int
        :return: game id, should be of length 8
        :rtype: str
        """
        if is_regular:
            return f'{season_year}02{str(game_number).zfill(4)}'
        else:
            return f'{season_year}03{str(game_number).zfill(4)}'


    def download_game(self, season_year: list, is_regular: bool, game_number: int, path_output="") -> dict:

        if path_output == "":
            if "NHL_DATA_DIR" in os.environ:
                path_output = os.environ["NHL_DATA_DIR"]
            else:
                print('NHL_DATA_DIR environment is not defined, please defined it before to continue.')
                return None

        if is_regular:
            path_output_season = os.path.join(path_output, str(season_year), "regular")
        else:
            path_output_season = os.path.join(path_output, str(season_year), "playoffs")

        os.makedirs(path_output_season, exist_ok=True)

        game_id = self.build_game_id(season_year, is_regular, game_number)
        game_id_path = os.path.join(path_output_season, f'{game_id}.json')

        # If the json has already been download, just read it and go the next one
        if os.path.exists(game_id_path):
            try:
                json_dict = json.load(open(game_id_path))
                return json_dict
            except json.JSONDecodeError: # if the json file is not valid, retrieve it from the API
                pass

        # If not, download and save the json
        url = self.get_url(f'{game_id}')
        r = requests.get(url)
        if r.status_code == 200:
            data_json = r.json()
            json.dump(data_json, open(game_id_path, "w"), indent=4)
            return data_json
        else:
            return None



    def download_data(self, seasons_year: list, is_regular: bool, path_output="") -> dict:
        """Download data of season "season_year/season_year+1" and put them in the directory specified by path_output.
           By default, this directory is be specified by the environment variable NHL_DATA_DIR.
           If they are already in the directory, they will be read, without being downloaded

        :param season_year: specific year in format XXXX
        :type season_year: int
        :param is_regular: True if regular season, False if playoffs
        :type is_regular: bool
        :param path_output: specific game number
        :type path_output: str
        :return: a dictionary that contains the data downloaded. The keys are the season year
        :rtype: dict
        """

        if path_output == "":
            if "NHL_DATA_DIR" in os.environ:
                path_output = os.environ["NHL_DATA_DIR"]
            else:
                print('NHL_DATA_DIR environment is not defined, please defined it before to continue.')
                return None


        pbar_season = tqdm(seasons_year, position=0)
        for season_year in pbar_season:

            if is_regular:
                path_output_season = os.path.join(path_output, str(season_year), "regular")
                pbar_season.set_description(f'Season {season_year} - Regular')
            else:
                path_output_season = os.path.join(path_output, str(season_year), "playoffs")
                pbar_season.set_description(f'Season {season_year} - Playoffs')

            os.makedirs(path_output_season, exist_ok=True)

            if not self.validate_season_year(season_year):
                print(f'Cannot download season {season_year}')
                continue

            self.nhl_data[season_year] = []
            game_numbers = self.get_number_of_games(season_year, is_regular)

            pbar_game = tqdm(game_numbers, position=1, leave=True)
            for game_number in pbar_game:
                pbar_game.set_description(f'Game {game_number}')

                # Build the game id and get the path to load/save the json file
                game_id = self.build_game_id(season_year, is_regular, game_number)
                game_id_path = os.path.join(path_output_season, f'{game_id}.json')

                # If the json has already been download, just read it and go the next one
                if os.path.exists(game_id_path):
                    try:
                        json_dict = json.load(open(game_id_path))
                        self.nhl_data[season_year].append(json_dict)
                        continue
                    except json.JSONDecodeError: # if the json file is not valid, retrieve it from the API
                        pass

                # If not, download and save the json
                url = self.get_url(f'{game_id}')
                r = requests.get(url)
                if r.status_code == 200:
                    data_json = r.json()
                    json.dump(data_json, open(game_id_path, "w"), indent=4)
                    self.nhl_data[season_year].append(data_json)

        return self.nhl_data



    def __add__(self, other_data):
        """Add a new season to the current object

        :param other_data: another instance that contains some data
        :type other_data: NHLDataManager
        :return: the object self with data added for 'other_data'
        :rtype: self
        """

        for keys in other_data.nhl_data.keys():

            # Si la clé (l'année) n'existe pas, on la prend entièrement de 'other_data'
            if keys not in self.nhl_data.keys():
                self.nhl_data[keys] = other_data.nhl_data[keys]
            # Si la clé (l'année) existe déjà, on rajoute simplement les nouvelles données non déjà présentes
            else:
                for data in other_data.nhl_data[keys]:
                    if data not in self.nhl_data[keys]:
                        self.nhl_data[keys].append(data)

        return self



def get_game_ids(season_year:int, season_type:str, path_data="") -> list:
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

    if path_data=="":
        path_data = os.environ["NHL_DATA_DIR"]

    if season_type.lower() == "regular":
        path_data = os.path.join(path_data, str(season_year), "Regular")
    elif (season_type.lower() == "playoff") | (season_type.lower() == "playoffs"):
        path_data = os.path.join(path_data, str(season_year), "playoffs")

    if not os.path.exists(path_data):
        data_manager = NHLDataManager(path_data)
        data_manager.download_data(seasons_year=[season_year], is_regular=(season_type.lower() == "regular"), path_output=path_data)

    json_files = os.listdir(path_data)
    json_files = [f for f in json_files if f.endswith('.json')]
    
    games_id = [int(f[0:8]) for f in json_files]

    return games_id


def get_game_numbers(season_year:int, season_type:str, path_data="") -> list:
    """Return the list of game number for a specific season year and type (regular or playoffs)

    :param season_year: specific season year
    :type season_year: int
    :param season_type: 'Regular' or 'Playoffs'
    :type season_type: str
    :param path_data: path where the data will be [read if exist] - [stored if not]
    :type path_data: str
    :return: the list of game number (3 or 4 digits)
    :rtype: list[int]
    """

    if path_data=="":
        path_data = os.environ["NHL_DATA_DIR"]

    if season_type.lower() == "regular":
        path_data = os.path.join(path_data, str(season_year), "Regular")
    elif (season_type.lower() == "playoff") | (season_type.lower() == "playoffs"):
        path_data = os.path.join(path_data, str(season_year), "playoffs")

    if not os.path.exists(path_data):
        data_manager = NHLDataManager(path_data)
        data_manager.download_data(seasons_year=[season_year], is_regular=(season_type.lower() == "regular"), path_output=path_data)

    json_files = os.listdir(path_data)
    json_files = [f for f in json_files if f.endswith('.json')]
    
    games_number = [int(f[6:10]) for f in json_files]

    return games_number


def get_teams_from_data(nhl_data : dict) -> dict:
    """Return the teams from the data

    :param nhl_data: data of a specific game already (down)loaded
    :type nhl_data: dict
    :return: a dictionary {abbr_home:name_home, abbr_away:name_away}
    :rtype: dict
    """
    try:
        team_name_away = nhl_data['gameData']['teams']['away']['name']
        team_abbr_away = nhl_data['gameData']['teams']['away']['abbreviation']

        team_name_home = nhl_data['gameData']['teams']['home']['name']
        team_abbr_home = nhl_data['gameData']['teams']['home']['abbreviation']

        return {team_abbr_home:team_name_home, team_abbr_away:team_name_away}
    except KeyError:
        return {}

def get_teams(season_year:int, season_type:str, game_number:int, path_data="") -> dict:

    """Return the teams of a specific game

    :param season_year: specific season year
    :type season_year: int
    :param season_type: 'Regular' or 'Playoffs'
    :type season_type: str
    :param game_num: specific game number (could be get from the get_game_numbers() function)
    :type game_num: int
    :param path_data: path where the data will be [read if exist] - [stored if not]
    :type path_data: str
    :return: a dictionary {abbr_home:name_home, abbr_away:name_away}
    :rtype: dict
    """

    data_manager = NHLDataManager()
    data = data_manager.download_game(season_year=season_year, is_regular=(season_type.lower() == "regular"), game_number=game_number, path_output=path_data)

    if data is None:
        return {}

    return get_teams_from_data(data)



def get_final_score_from_data(nhl_data : dict) -> dict:
    """Return the final score of a specific game

    :param nhl_data: data of a specific game already (down)loaded
    :type nhl_data: dict
    :return: a dictionary {abbr_home:score, abbr_away:score}
    :rtype: dict
    """

    try:
        team_abbr_away = nhl_data['gameData']['teams']['away']['abbreviation']
        team_abbr_home = nhl_data['gameData']['teams']['home']['abbreviation']

    except KeyError:
        return {}


    try:
        score_away = nhl_data['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats']['goals']
        score_home = nhl_data['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats']['goals']

        
    except KeyError:
        score_away = 0
        score_home = 0

    return {team_abbr_home:score_home, team_abbr_away:score_away}


def get_final_score(season_year:int, season_type:str, game_number:int, path_data="") -> dict:
    """Return the final score of a specific game

    :param season_year: specific season year
    :type season_year: int
    :param season_type: 'Regular' or 'Playoffs'
    :type season_type: str
    :param game_num: specific game number (could be get from the get_game_numbers() function)
    :type game_num: int
    :param path_data: path where the data will be [read if exist] - [stored if not]
    :type path_data: str
    :return: a dictionary {abbr_home:score, abbr_away:score}
    :rtype: dict
    """

    data_manager = NHLDataManager()
    data = data_manager.download_game(season_year=season_year, is_regular=(season_type.lower() == "regular"), game_number=game_number, path_output=path_data)

    if data is None:
        return {}

    return get_final_score_from_data(data)




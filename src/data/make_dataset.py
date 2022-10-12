# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import time

import NHLDataManager


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def download_data():
    data_manager = NHLDataManager.NHLDataManager()
    seasons_year = [2016, 2017]
    data_manager.download_data(seasons_year=seasons_year, seaons_type="Regular")
    data_manager.download_data(seasons_year=seasons_year, season_type="Playoffs")

def create_df(season_year, season_type, game_number):
    data_manager = NHLDataManager.NHLDataManager()
    df = data_manager.get_goals_and_shots_df(season_year, season_type, game_number)
    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # time_start = time.time()
    # download_data()
    # time_end = time.time()
    # print(f'Time spent to download the data: {time_end-time_start}')

    season_year, season_type, game_number = 2016, "Regular", 428
    df = create_df(season_year, season_type, game_number)

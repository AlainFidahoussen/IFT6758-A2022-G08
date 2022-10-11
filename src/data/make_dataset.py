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
    nhl_data_regular = data_manager.download_data(seasons_year=seasons_year, is_regular=True)
    nhl_data_playoffs = data_manager.download_data(seasons_year=seasons_year, is_regular=False)

    nhl_data = {}
    nhl_data['Regular'] = nhl_data_regular
    nhl_data['Playoffs'] = nhl_data_playoffs

    return nhl_data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    time_start = time.time()
    nhl_data = download_data()
    time_end = time.time()
    print(f'Time spent to download the data: {time_end-time_start}')

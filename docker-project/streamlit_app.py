import os
import sys

sys.path.append(r"C:\Users\anniw\IFT6758-A2022-G08\docker-project\ift6758\ift6758\client")

from dotenv import load_dotenv
load_dotenv();

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

import serving_client
import game_client


st.title("Hockey Visualization App")

with st.sidebar:
    # TODO: Add input for the sidebar
    workspace = st.selectbox('Workspace', ['ift6758-a22-g08'])
    model = st.selectbox('Model', ['xgboost-randomforest-ii', 'randomforest-allfeatures'])
    version = st.selectbox('Version', ['1.0.0'])
    model_button = st.button('Get model')
    
    if model_button: # True is clicked, else False
        st.write(f'{workspace} + {model} + {version}')
        st.session_state['model_downloaded'] = True # Like a dictionary
        sc = serving_client.ServingClient()
        sc.download_registry_model(workspace, model, version)
    

with st.container():
    # TODO: Add Game ID input
    # game_id = st.selectbox('Game ID', options=['2021020329', '2021020330', '2021011229'])
    game_id = st.text_input('Game ID', '2016020020')
    ping_button = st.button('Ping game')

    if ping_button:
        # code pour faire des requetes
        season_year = int(game_id[:4])
        if game_id[4:6] == '02':
            season_type = 'Regular'
        elif game_id[4:6] == '03':
            season_type = 'Playoffs'
        game_number = int(game_id[6:].lstrip('0'))

        gc = game_client.GameClient()
        df_features = gc.ping_game(season_year, season_type, game_number)
        # df_features_out = sc.predict(df_features)
        

st.markdown('')

with st.container():
    # TODO: Add Game info and predictions
    if ping_button:
        game_url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"
        r = requests.get(game_url)
        if r.status_code == 200:
            data = r.json()

        home_team = data['gameData']['teams']['home']['name']
        home_team_abb = data['gameData']['teams']['home']['abbreviation']
        away_team = data['gameData']['teams']['away']['name']
        away_team_abb = data['gameData']['teams']['away']['abbreviation']
        st.subheader(f'Game {game_id} : {home_team} vs {away_team}')

        period = data['liveData']['linescore']['currentPeriod']
        time_remaining = data['liveData']['linescore']['currentPeriodTimeRemaining']
        st.text(f'Period {period} - {time_remaining} left')

        col1, col2 = st.columns(2)
        home_actual_goal = int(data['liveData']['plays']['currentPlay']['about']['goals']['home'])
        away_actual_goal = int(data['liveData']['plays']['currentPlay']['about']['goals']['away'])
        # full_home_name = f'{home_team} ({home_team_abb})'
        # full_away_name = f'{away_team} ({away_team_abb})'
        # home_predict_goal = df_features_out[df_features_out['Team'] == full_home_name]['Shot probability'].sum() 
        # away_predict_goal = df_features_out[df_features_out['Team'] == full_away_name]['Shot probability'].sum() 
        # home_diff = str(home_predict_goal - home_actual_goal)
        # away_diff = str(away_predict_goal - away_actual_goal)

        # Test  
        home_predict_goal = 3.2 
        away_predict_goal = 1.4
        home_diff = str(round(home_predict_goal - home_actual_goal, 2))
        away_diff = str(round(away_predict_goal - away_actual_goal, 2))

        col1.metric(f'{home_team} xG (actual)', f'{home_predict_goal} ({home_actual_goal})', home_diff)
        col1.metric(f'{away_team} xG (actual)', f'{away_predict_goal} ({away_actual_goal})', away_diff)


st.markdown('')

with st.container():
    # TODO: Add data used for predictions
    if ping_button:
        st.subheader('Data used for predictions (and predictions)')
        # df = pd.DataFrame(np.random.randn(50,20), 
        #                  columns=('col %d' % i for i in range(20)))
        st.dataframe(df_features)
    
st.markdown('')

with st.container(): # TODO: Bonus
    power = st.slider('What power?', 0, 10, 1)
    fig, ax = plt.subplots()
    data = [i ** power for i in range(10)]
    ax.plot(data)
    st.pyplot(fig)

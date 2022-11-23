from dotenv import load_dotenv
load_dotenv();

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager


import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.ensemble import GradientBoostingClassifier

from imblearn.pipeline import Pipeline
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from comet_ml import Experiment
from comet_ml import Optimizer

def start_experiment():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file containing the api key
        project_name="milestone-2",
        workspace="ift6758-a22-g08",
    )
    return experiment


def evaluate(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'macro f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
    }

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def GetData():

    # Get the dataset
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    data_df = FeaturesManager.build_features(seasons_year, season_type, with_player_stats=True, with_strength_stats=True)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()

    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)

    df_features = data_df[feature_names]
    df_targets = data_df[target_name]

    X = df_features
    y = df_targets

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    X_train, y_train = OutliersManager.remove_outliers(X_train, y_train)
    X_valid, y_valid = OutliersManager.remove_outliers(X_valid, y_valid)

    X_train['Rebound'] = ((X_train['Rebound'] == 1) & (X_train['Last event elapsed time'] < 4)).astype(int)
    X_valid['Rebound'] = ((X_valid['Rebound'] == 1) & (X_valid['Last event elapsed time'] < 4)).astype(int)

    distance_bins = np.linspace(0,185,10)
    angle_bins = np.linspace(-185,185,10)
    X_train['Angle Bins'] = pd.cut(X_train['Shot angle'], bins=angle_bins, include_lowest=True, labels=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8'])
    X_train['Distance Bins'] = pd.cut(X_train['Shot distance'], bins=distance_bins, include_lowest=True, labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'] )

    X_valid['Angle Bins'] = pd.cut(X_valid['Shot angle'], bins=angle_bins, include_lowest=True, labels=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8'])
    X_valid['Distance Bins'] = pd.cut(X_valid['Shot distance'], bins=distance_bins, include_lowest=True, labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'] )

    X_train.drop(labels=['Shot angle', 'Shot distance'], axis=1)

    return X_train, X_valid, y_train, y_valid


def run_search(experiment, model, X, y, cv):
  # fit the model on the whole dataset
  results = cross_validate(
      model, X, y, cv=cv, 
      scoring=[
          "f1_macro", 
          "precision_macro",  
          "recall_macro",
          "roc_auc"
      ], return_train_score=True)

  for k in results.keys():
    scores = results[k]
    for idx, score in enumerate(scores):
      experiment.log_metrics({f"cv_{k}": score}, step=idx)

    experiment.log_metrics({f"cv_mean_{k}": np.mean(scores)})
    experiment.log_metrics({f"cv_std_{k}": np.std(scores)})

    experiment.log_parameter("random_state", RANDOM_SEED)

def GradientBoostHyperParameters(project_name: str):

    # numerical_columns = [
    #     'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
    #     'Speed From Previous Event', 'Change in Shot Angle', 
    #     'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
    #     'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
    #     'Last event distance', 'Last event angle']

    # nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position']
    # ordinal_columns = ['Period', 'Num players With', 'Num players Against', 'Is Empty', 'Rebound']

    numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']

    nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position', 'Angle Bins', 'Distance Bins']
    ordinal_columns = ['Period', 'Num players With', 'Num players Against', 'Is Empty', 'Rebound']


    # median
    fill_nan = ColumnTransformer(transformers = [
        ('cat', SimpleImputer(strategy ='most_frequent'), nominal_columns + ordinal_columns),
        ('num', SimpleImputer(strategy ='median'), numerical_columns),
    ], remainder ='passthrough')

    # one-hot      
    one_hot = ColumnTransformer(transformers = [
        ('enc', OneHotEncoder(sparse = False), list(range(len(nominal_columns)))),
    ], remainder ='passthrough')


    # setting the spec for bayes algorithm
    spec = {
        "objective": "maximize",
        "metric": "cv_mean_test_f1_macro",
        "seed": RANDOM_SEED
    }

    # setting the parameters we are tuning
    model_params = {
        "n_estimators": {
            "type": "integer",
            "scaling_type": "uniform",
            "min": 50,
            "max": 150},
        "max_features": {
            "type": "categorical",
            "values": ["sqrt", "log2"]},
        "max_depth": {
            "type": "discrete",
            "values": [2, 5, 8, 12, 15, 20]},
        "learning_rate": {
            "type": "discrete",
            "values": [0.2, 0.4, 0.6, 0.8]}
        # "selector_n_estimators": {
        #     "type": "integer",
        #     "scaling_type": "uniform",
        #     "min": 20,
        #     "max": 100}
    }

    # defining the configuration dictionary
    config_dict = {
        "algorithm": "bayes",
        "spec": spec, 
        "parameters": model_params,
        "name": "Bayes Optimization", 
        "trials": 5
    }

    cv = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)

    # initializing the comet ml optimizer
    opt = Optimizer(
        api_key=os.environ.get('COMET_API_KEY'),
        config=config_dict,
        project_name=project_name,
        workspace="ift6758-a22-g08")

    X_train, X_valid, y_train, y_valid = GetData()

    for experiment in opt.get_experiments():

        n_estimators          = experiment.get_parameter("n_estimators")
        max_features          = experiment.get_parameter("max_features")
        max_depth             = experiment.get_parameter("max_depth")
        learning_rate         = experiment.get_parameter("learning_rate")
        # selector_n_estimators = experiment.get_parameter("selector_n_estimators")

        clf_gradientboost = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=RANDOM_SEED)
        
        # selector = FeaturesSelector.SelectFromRandomForest(selector_n_estimators)

        # Pipeline
        steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ("clf_gradientboost", clf_gradientboost)]
        pipeline = Pipeline(steps=steps)

        run_search(experiment, pipeline, X_train, y_train, cv)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)

        experiment.end()
  

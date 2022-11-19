from dotenv import load_dotenv
load_dotenv();

import src.visualization.visualize as VizManager
import src.data.NHLDataManager as DataManager
import src.features.build_features as FeaturesManager

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

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

    names = ['Period', 'Period seconds', 'Shot Type', 'Shot distance', 'Shot angle', 'Is Empty',
             'Strength', 'Rebound', 'Speed From Previous Event', 'Change in Shot Angle', 
             'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season', 
             'Num players With', 'Num players Against', 'Elapsed time since Power Play',
             'Last event elapsed time', 'Last event distance', 'Last event angle', 
             'Is Goal']

    feature_names, target_name = names[0:-2], names[-1]
    feature_names = np.array(feature_names)

    df_features = data_df[feature_names]
    df_targets = data_df[target_name]

    df_features = df_features.fillna(df_features.median())
    df_features['Shot Type'] = df_features['Shot Type'].fillna(df_features['Shot Type'].mode().iloc[0])
    df_features['Shot angle'] = df_features['Shot angle'].abs()


    dummy_shot_type = pd.get_dummies(df_features['Shot Type'], prefix='Shot Type')
    df_features = df_features.merge(dummy_shot_type, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Shot Type'])

    dummy_strength = pd.get_dummies(df_features['Strength'], prefix='Strength')
    df_features = df_features.merge(dummy_strength, left_index=True, right_index=True)
    df_features = df_features.drop(columns=['Strength'])

    # Update features_name
    feature_names = list(df_features.columns)
    feature_names = np.array(feature_names)
    X = df_features
    y = df_targets

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    return X_train, X_valid, y_train, y_valid



def RandomForestHyperParameters():

    # setting the spec for bayes algorithm
    spec = {
        "objective": "minimize",
        "metric": "loss",
        "seed": RANDOM_SEED
    }

    # setting the parameters we are tuning
    model_params = {
        "n_estimators": {
            "type": "integer",
            "scaling_type": "uniform",
            "min": 100,
            "max": 300},
        "criterion": {
            "type": "categorical",
            "values": ["gini", "entropy"]},
        "max_depth": {
            "type": "discrete",
            "values": [5, 10, 15, 20]},
        "sampling_strategy": {
            "type": "discrete",
            "values": [0.5, 0.6, 0.7, 0.8, 0.9]
        }
    }

    # defining the configuration dictionary
    config_dict = {
        "algorithm": "bayes",
        "spec": spec, 
        "parameters": model_params,
        "name": "Bayes Optimization", 
        "trials": 5
    }


    # initializing the comet ml optimizer
    opt = Optimizer(
        api_key=os.environ.get('COMET_API_KEY'),
        config=config_dict,
        project_name="Hyperparameters-RandomForest",
        workspace="ift6758-a22-g08")

    X_train, X_valid, y_train, y_valid = GetData()

    for experiment in opt.get_experiments():

        n_estimators      = experiment.get_parameter("n_estimators")
        criterion         = experiment.get_parameter("criterion")
        max_depth         = experiment.get_parameter("max_depth")
        sampling_strategy = experiment.get_parameter("sampling_strategy")

        random_forest = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_SEED)

        scaler = StandardScaler()
        steps = [('s', scaler), ("Balanced Random Forest", random_forest)]
        pipeline = Pipeline(steps=steps)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_valid)
        metrics = evaluate(y_valid, y_pred)

        experiment.log_metrics(metrics)
        experiment.log_confusion_matrix(y_valid.to_numpy().astype(int), y_pred.astype(int))
        experiment.log_parameter("random_state", RANDOM_SEED)

        experiment.end()
  

if __name__ == "__main__":
    RandomForestHyperParameters()

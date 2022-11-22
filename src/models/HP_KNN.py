from dotenv import load_dotenv
load_dotenv();

import numpy as np
import os
import sys 

script_dir = os.path.dirname( __file__ )
module_dir = os.path.join(script_dir, '../..')
sys.path.append(module_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.neighbors import KNeighborsClassifier

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager

from imblearn.pipeline import Pipeline
from imblearn.ensemble import EasyEnsembleClassifier
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from comet_ml import Experiment
from comet_ml import Optimizer


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

    return X_train, X_valid, y_train, y_valid


def KNNParameters():
    
    numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']

    nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position']
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

    # scaler
    scaler = StandardScaler()

    # features selectpr
    selector = FeaturesSelector.SelectFromLinearSVC()
    
    
    # setting the spec for bayes algorithm
    spec = {
        "objective": "minimize",
        "metric": "loss",
        "seed": RANDOM_SEED
    }

    # setting the parameters we are tuning
    model_params = {
        "n_neighbors": {
            "type": "discrete",
            "values": [5, 7, 9, 11, 13, 15]
        },
        "weights": {
            "type": "categorical",
            "values": ["uniform", "distance"]
        },
        "algorithm": {
            "type": "categorical",
            "values": ["auto", "ball_tre", "kd_tree", "brute"]
        },
    }

    # defining the configuration dictionary
    config_dict = {
        "algorithm": "bayes",
        "spec": spec, 
        "parameters": model_params,
        "name": "Bayes Optimization", 
        "trials": 3
    }


    # initializing the comet ml optimizer
    opt = Optimizer(
        api_key=os.environ.get('COMET_API_KEY'),
        config=config_dict,
        project_name="hyperparameters-KNN-2",
        workspace="ift6758-a22-g08")

    X_train, X_valid, y_train, y_valid = GetData()

    for experiment in opt.get_experiments():

        n_neighbors      = experiment.get_parameter("n_neighbors")
        weights = experiment.get_parameter("weights")
        algorithm = experiment.get_parameter("algorithm")

        clf_KNN = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm)

        scaler = StandardScaler()

        # Pipeline
        steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ('scaler', scaler), ('selector', selector), ("clf_KNN", clf_KNN)]
        pipeline = Pipeline(steps=steps)

        pipeline.fit(X_train, y_train)

        # with experiment.train():
        #     y_pred = pipeline.predict(X_train)
        #     metrics = evaluate(y_train, y_pred)
        #     experiment.log_metrics(metrics)
        #     experiment.log_confusion_matrix(y_train.to_numpy().astype(int), y_pred.astype(int))

        with experiment.validate():
            y_pred = pipeline.predict(X_valid)
            metrics = evaluate(y_valid, y_pred)
            experiment.log_metrics(metrics)
            experiment.log_confusion_matrix(y_valid.to_numpy().astype(int), y_pred.astype(int))
        
        experiment.log_parameter("random_state", RANDOM_SEED)
        experiment.end()
        
if __name__ == "__main__":
    KNNParameters()
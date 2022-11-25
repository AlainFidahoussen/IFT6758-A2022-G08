import os
import sys

# Only for me
script_dir = os.path.dirname( __file__ )
module_dir = os.path.join(script_dir, '../..')
sys.path.append(module_dir)

# To load the environment variable defined in the .env file
from dotenv import load_dotenv
load_dotenv();

# import comet_ml at the top of your file
from comet_ml import Experiment
from comet_ml import API

import numpy as np
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import src.visualization.visualize as VizManager
import src.features.build_features as FeaturesManager

import pickle

RANDOM_SEED = 42


def start_experiment():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file containing the api key
        project_name="Logistic Regression",
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

def GetTrainingData():

    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)

    # We take the absolute value, for symmetry reasons
    features_data['Shot angle'] = features_data['Shot angle'].abs()

    X = features_data[['Shot distance', 'Shot angle']]
    y = features_data['Is Goal']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    return  X_train, X_valid, y_train, y_valid


def DoTraining():

    X_train, X_valid, y_train, y_valid = GetTrainingData()
    
    clf_distance(X_train, X_valid, y_train, y_valid, RANDOM_SEED)
    clf_angle(X_train, X_valid, y_train, y_valid, RANDOM_SEED)
    clf_distance_angle(X_train, X_valid, y_train, y_valid, RANDOM_SEED)


def DoTesting(season_year, season_type):

    features_data = FeaturesManager.build_features([season_year], season_type)

    # We take the absolute value, for symmetry reasons
    features_data['Shot angle'] = features_data['Shot angle'].abs()

    y_test = features_data['Is Goal']

    api = API()

    workspace_name = "ift6758-a22-g08"

    # Download and evaluate the Logistic Regresion on Distance
    api.download_registry_model(workspace_name, "logisticregression-distance", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "LogisticRegression_distance.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    X_test = features_data[['Shot distance']]
    y_pred = clf.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    
    print('--------------------------------')
    print('Logistic Regression - Distance')
    print(metrics)


    # Download and evaluate the Logistic Regresion on Angle
    api.download_registry_model(workspace_name, "logisticregression-angle", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "LogisticRegression_angle.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    X_test = features_data[['Shot angle']]
    y_pred = clf.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    
    print('--------------------------------')
    print('Logistic Regression - Angle')
    print(metrics)


    # Download and evaluate the Logistic Regresion on Distance and Angle
    api.download_registry_model(workspace_name, "logisticregression-distance-angle", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "LogisticRegression_distance_angle.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    X_test = features_data[['Shot distance', 'Shot angle']]
    y_pred = clf.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    
    print('--------------------------------')
    print('Logistic Regression - Distance/Angle')
    print(metrics)


def clf_distance(X_train, X_valid, y_train, y_valid, RANDOM_SEED):
    experiment = start_experiment()
    
    experiment.set_name('LogisticRegression_distance')
    X_distance_train, X_distance_valid = X_train[['Shot distance']], X_valid[['Shot distance']]
    experiment.log_dataset_hash(X_distance_train)
    
    clf_distance = LogisticRegression(random_state=RANDOM_SEED).fit(X_distance_train, y_train)
    
    with experiment.train():
        y_distance_pred_train = clf_distance.predict(X_distance_train)
        metrics = evaluate(y_train, y_distance_pred_train)
        experiment.log_metrics(metrics)
    
    pkl_filename = './models/LogisticRegression_distance.pkl'
    # pkl_filename = 'LogisticRegression_distance.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_distance, file)
    experiment.log_model("LogisticRegression_distance", pkl_filename)
    experiment.register_model("LogisticRegression_distance")

    with experiment.validate():
        y_distance_pred = clf_distance.predict(X_distance_valid)
        metrics = evaluate(y_valid, y_distance_pred)
        experiment.log_metrics(metrics)
        
    params={"random_state": RANDOM_SEED,
        "model_type": "logreg",
        "scaler": None,
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Shot distance",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
def clf_angle(X_train, X_valid, y_train, y_valid, RANDOM_SEED):
    experiment = start_experiment()
    
    experiment.set_name('LogisticRegression_angle')
    X_angle_train, X_angle_valid = X_train[['Shot angle']], X_valid[['Shot angle']]
    experiment.log_dataset_hash(X_angle_train)
    
    clf_angle = LogisticRegression(random_state=RANDOM_SEED).fit(X_angle_train, y_train)
    
    with experiment.train():
        y_angle_pred_train = clf_angle.predict(X_angle_train)
        metrics = evaluate(y_train, y_angle_pred_train)
        experiment.log_metrics(metrics)
        
    pkl_filename = './models/LogisticRegression_angle.pkl'
    # pkl_filename = 'LogisticRegression_angle.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_angle, file)
    experiment.log_model("LogisticRegression_angle", pkl_filename)
    experiment.register_model("LogisticRegression_angle")

    with experiment.validate():
        y_angle_pred = clf_angle.predict(X_angle_valid)
        metrics = evaluate(y_valid, y_angle_pred)
        experiment.log_metrics(metrics)
        
    params={"random_state": RANDOM_SEED,
        "model_type": "logreg",
        "scaler": None,
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Shot angle",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
def clf_distance_angle(X_train, X_valid, y_train, y_valid, RANDOM_SEED):
    experiment = start_experiment()
    experiment.set_name('LogisticRegression_distance_angle')
    X_distance_angle_train, X_distance_angle_valid = X_train, X_valid
    experiment.log_dataset_hash(X_distance_angle_train)
    
    clf_distance_angle = LogisticRegression(random_state=RANDOM_SEED).fit(X_distance_angle_train, y_train)
    
    with experiment.train():
        y_distance_angle_pred_train = clf_distance_angle.predict(X_distance_angle_train)
        metrics = evaluate(y_train, y_distance_angle_pred_train)
        experiment.log_metrics(metrics)
        
    pkl_filename = './models/LogisticRegression_distance_angle.pkl'
    # pkl_filename = 'LogisticRegression_distance_angle.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_distance_angle, file)
    experiment.log_model("LogisticRegression_distance_angle", pkl_filename)
    experiment.register_model("LogisticRegression_distance_angle")

    with experiment.validate():
        y_distance_angle_pred = clf_distance_angle.predict(X_distance_angle_valid)
        metrics = evaluate(y_valid, y_distance_angle_pred)
        experiment.log_metrics(metrics)
        
    params={"random_state": RANDOM_SEED,
        "model_type": "logreg",
        "scaler": None,
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Shot distance and angle",}
    experiment.log_parameters(params)
        
    experiment.end()

    
   

if __name__ == "__main__":
    # DoTraining()
    DoTesting(2019, "Regular")
    DoTesting(2019, "Playoffs")


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

from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager

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

    numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']

    nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position']
    ordinal_columns = ['Period', 'Num players With', 'Num players Against', 'Is Empty', 'Rebound']
    
    # Get the dataset
    X_train, X_valid, y_train, y_valid = FeaturesManager.GetTrainValid_II()

    X_train, y_train = OutliersManager.remove_outliers(X_train, y_train)
    X_valid, y_valid = OutliersManager.remove_outliers(X_valid, y_valid)
    
    select_from_PCA = FeaturesSelector.SelectFromPCA(9)
    select_from_PCA.fit(X_train, y_train)

    X_train, y_train = select_from_PCA.transform(X_train, y_train)
    X_valid, y_valid = select_from_PCA.transform(X_valid, y_valid)


    return X_train, X_valid, y_train, y_valid



def SVCParameters():
    
    # setting the spec for bayes algorithm
    spec = {
        "objective": "minimize",
        "metric": "loss",
        "seed": RANDOM_SEED
    }

    # setting the parameters we are tuning
    model_params = {
        "C": {
            "type": "integer",
            "scaling_type": "uniform",
            "min": 0.01,
            "max": 100
        },
        "kernel": {
            "type": "categorical",
            "values": ["linear", "poly", "rbf", "sigmoid"]
        },
        "degree": {
            "type": "discrete",
            "values": [2, 3, 4, 5]
        },
        "gamma": {
            "type": "integer",
            "scaling_type": "uniform",
            "min": 0.001,
            "max": 1
        },
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
        project_name="hyperparameters-svc",
        workspace="ift6758-a22-g08")

    X_train, X_valid, y_train, y_valid = GetData()

    for experiment in opt.get_experiments():

        C      = experiment.get_parameter("C")
        kernel = experiment.get_parameter("kernel")
        degree = experiment.get_parameter("degree")
        gamma  = experiment.get_parameter("gamma")

        clf_SVC = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            random_state=RANDOM_SEED)

        # Pipeline
        steps = [("clf_SVC", clf_SVC)]
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
    SVCParameters()
from dotenv import load_dotenv
load_dotenv();

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager


import numpy as np
import os
import pandas as pd
import cloudpickle


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from comet_ml import Experiment
from comet_ml import API

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def GetTrainingData():
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)
    return features_data

    
class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self._features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._features]
    


def start_experiment():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file containing the api key
        project_name="best-models-calibrated",
        workspace="ift6758-a22-g08",
    )
    return experiment


def evaluate(y_true, y_proba):
    y_pred = np.round(y_proba)
    return {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'macro f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'rocauc': roc_auc_score(y_true, y_proba)
    }



def DoTraining(calibrated=False):
    training_data = GetTrainingData()
    clf_randomforest(training_data, calibrated)


def clf_randomforest(training_data, calibrated=False):
    experiment = start_experiment()
    
    experiment.set_name('RandomForest_AllFeatures')

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

    # columns selector
    columns_selector = ColumnsSelector(numerical_columns + nominal_columns + ordinal_columns)

    experiment.log_dataset_hash(training_data)

    X, y = training_data.drop(labels=['Is Goal'], axis=1), training_data['Is Goal']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
  
    X_train, y_train = OutliersManager.remove_outliers(X_train, y_train)

    # Classifier
    n_estimators = 82
    max_depth = 10
    sampling_strategy = 0.5
    min_samples_split = 8
    criterion = 'entropy'
    max_features = 'log2'

    clf_forest = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        sampling_strategy=sampling_strategy,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_features=max_features,
        random_state=RANDOM_SEED)
    
    # Pipeline
    if calibrated:
        steps = [("columns_selector", columns_selector), 
                 ('fill_nan', fill_nan), 
                 ('one_hot', one_hot), 
                 ("clf_forest",  CalibratedClassifierCV(base_estimator=clf_forest, cv=3))]
    else:
        steps = [("columns_selector", columns_selector), 
                 ('fill_nan', fill_nan), 
                 ('one_hot', one_hot), 
                 ("clf_forest", clf_forest)]

    pipeline = Pipeline(steps=steps).fit(X_train, y_train)
    
    if calibrated:
        filename = 'RandomForest_AllFeatures_calibrated'
    else:
        filename = 'RandomForest_AllFeatures'
    pkl_filename = os.path.join(os.environ['NHL_MODEL_DIR'], filename + '.pkl')
    
    with open(pkl_filename, 'wb') as file:
        cloudpickle.dump(pipeline, file)
    experiment.log_model(filename, pkl_filename)
    experiment.register_model(filename)

    with experiment.train():
        y_proba = pipeline.predict_proba(X_train)[:,1]
        metrics = evaluate(y_train, y_proba)
        experiment.log_metrics(metrics)
        experiment.log_confusion_matrix(y_train.to_numpy().astype(int), np.around(y_proba).astype(int))

    with experiment.validate():
        y_proba = pipeline.predict_proba(X_valid)[:,1]
        metrics = evaluate(y_valid, y_proba)
        experiment.log_metrics(metrics)
        experiment.log_confusion_matrix(y_valid.to_numpy().astype(int), np.around(y_proba).astype(int))
        
    params={"random_state": RANDOM_SEED,
        "model_type": "forest",
        "scaler": "False",
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "AllFeatures",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
   

if __name__ == "__main__":
    DoTraining(calibrated=True)
    # DoTesting(2019, "Regular")
    # DoTesting(2019, "Playoffs")




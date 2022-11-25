from dotenv import load_dotenv
load_dotenv();

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager


import numpy as np
import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from comet_ml import Experiment
from comet_ml import Optimizer
from comet_ml import API

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def process_data(X, y):

    X, y = OutliersManager.remove_outliers(X, y)
    X['Rebound'] = ((X['Rebound'] == 1) & (X['Last event elapsed time'] < 4)).astype(int)
    
    distance_bins = np.linspace(0,185,10)
    angle_bins = np.linspace(-185,185,10)
    X['Angle Bins'] = pd.cut(X['Shot angle'], bins=angle_bins, include_lowest=True, labels=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8'])
    X['Distance Bins'] = pd.cut(X['Shot distance'], bins=distance_bins, include_lowest=True, labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'] )

    X.drop(labels=['Shot angle', 'Shot distance'], axis=1)
    return X, y


def GetTrainingData(shap=False):
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X = features_data[feature_names]
    y = features_data[target_name]

    if shap:
        X, y = process_data(X, y)
        numerical_columns = [
            'Elapsed time since Power Play', 'Last event elapsed time', 
            'st_Y', 'Last event angle', 'Change in Shot Angle']

        nominal_columns = ['Strength', 'Rebound', 'Angle Bins', 'Distance Bins']
        ordinal_columns = ['Num players Against']

    else:
        numerical_columns = [
            'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
            'Speed From Previous Event', 'Change in Shot Angle', 
            'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
            'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
            'Last event distance', 'Last event angle']

        nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position']
        ordinal_columns = ['Period', 'Num players With', 'Num players Against', 'Is Empty', 'Rebound']


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    X_train = X_train[numerical_columns + ordinal_columns + nominal_columns]
    X_valid = X_valid[numerical_columns + ordinal_columns + nominal_columns]
    
    return X_train, X_valid, y_train, y_valid, numerical_columns, nominal_columns, ordinal_columns


def GetTestingData(season_year, season_type):
    features_data = FeaturesManager.build_features([season_year], season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X_test = features_data[feature_names]
    y_test = features_data[target_name]

    numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']

    nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position']
    ordinal_columns = ['Period', 'Num players With', 'Num players Against', 'Is Empty', 'Rebound']

    return X_test, y_test, numerical_columns, nominal_columns, ordinal_columns


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


def RandomForestHyperParameters(project_name: str):

    X_train, X_Valid, y_train, y_valid, numerical_columns, nominal_columns, ordinal_columns = GetTrainingData()
   
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
            "max": 200},
        "max_depth": {
            "type": "discrete",
            "values": [5, 10, 15, 20]},
        "min_samples_split": {
            "type": "discrete",
            "values": [2, 4, 6, 8]},
        "criterion": {
            "type": "categorical",
            "values": ['gini', 'entropy']},
        "max_features": {
            "type": "categorical",
            "values": ['sqrt', 'log2']},
        "sampling_strategy": {
            "type": "discrete",
            "values": [0.3, 0.4, 0.5, 0.6]},
        "class_weight": {
            "type": "categorical",
            "values": ["balanced", "balanced_subsample"]}
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

    for experiment in opt.get_experiments():

        n_estimators        = experiment.get_parameter("n_estimators")
        max_depth           = experiment.get_parameter("max_depth")
        sampling_strategy   = experiment.get_parameter("sampling_strategy")
        min_samples_split   = experiment.get_parameter("min_samples_split")
        criterion           = experiment.get_parameter("criterion")
        max_features        = experiment.get_parameter("max_features")
        class_weight        = experiment.get_parameter("class_weight")

        clf_forest = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            sampling_strategy=sampling_strategy,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=max_features,
            random_state=RANDOM_SEED)

        # Pipeline
        steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ("clf_forest", clf_forest)]
        pipeline = Pipeline(steps=steps)

        run_search(experiment, pipeline, X_train, y_train, cv)

        experiment.end()
  


# ---------------------------------------------------#
# -------------------- BEST MODEL ------------------ #
# ---------------------------------------------------#
        

def start_experiment():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file containing the api key
        project_name="best-models",
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



def DoTraining():
    X_train, X_valid, y_train, y_valid, numerical_columns, nominal_columns, ordinal_columns = GetTrainingData()
    clf_randomforest(X_train, X_valid, y_train, y_valid, numerical_columns, nominal_columns, ordinal_columns)


def DoTesting(season_year, season_type):

    X_test, y_test, _ = GetTestingData(season_year, season_type)
    api = API()

    workspace_name = "ift6758-a22-g08"

    # Download and evaluate the Logistic Regresion on Distance
    api.download_registry_model(workspace_name, "RandomForest", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "RandomForest.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    y_proba = clf.predict_proba(X_test)[:,1]
    metrics = evaluate(y_test, y_proba)
    
    print('--------------------------------')
    print('RandomForest - All Features')
    print(metrics)



def clf_randomforest(X_train, X_valid, y_train, y_valid, numerical_columns, nominal_columns, ordinal_columns):
    experiment = start_experiment()
    
    experiment.set_name('RandomForest')

    # median
    fill_nan = ColumnTransformer(transformers = [
        ('cat', SimpleImputer(strategy ='most_frequent'), nominal_columns + ordinal_columns),
        ('num', SimpleImputer(strategy ='median'), numerical_columns),
    ], remainder ='passthrough')

    # one-hot      
    one_hot = ColumnTransformer(transformers = [
        ('enc', OneHotEncoder(sparse = False), list(range(len(nominal_columns)))),
    ], remainder ='passthrough')


    experiment.log_dataset_hash(X_train)

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
    steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ("clf_forest", clf_forest)]
    pipeline = Pipeline(steps=steps).fit(X_train, y_train)
    

    pkl_filename = './models/RandomForest.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pipeline, file)
    experiment.log_model("RandomForest", pkl_filename)
    experiment.register_model("RandomForest")

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
        "data": "Binning",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
   

if __name__ == "__main__":
    DoTraining()
    # DoTesting(2019, "Regular")
    # DoTesting(2019, "Playoffs")




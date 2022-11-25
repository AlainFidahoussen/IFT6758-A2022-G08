from dotenv import load_dotenv
load_dotenv();

import numpy as np
import os
import sys 
import pickle

script_dir = os.path.dirname( __file__ )
module_dir = os.path.join(script_dir, '../..')
sys.path.append(module_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager

from imblearn.pipeline import Pipeline
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel

from comet_ml import Experiment
from comet_ml import Optimizer
from comet_ml import API


def evaluate(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'macro f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
    }

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def GetTrainingData():

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

def GetTestingData(season_year, season_type):

    # Get the dataset
    data_df = FeaturesManager.build_features([seasons_year], season_type, with_player_stats=True, with_strength_stats=True)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()

    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)

    df_features = data_df[feature_names]
    df_targets = data_df[target_name]

    X_test = df_features
    y_test = df_targets

    X_test, y_test = OutliersManager.remove_outliers(X_test, y_test)

    return X_test, y_test

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

    X_train, X_valid, y_train, y_valid = GetTrainingData()

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
        

# ---------------------------------------------------#
# -------------------- BEST MODEL ------------------ #
# ---------------------------------------------------#

def start_experiment():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file containing the api key
        project_name="Best-Models",
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
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X = features_data[feature_names]
    y = features_data[target_name]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    X_train, y_train = OutliersManager.remove_outliers(X_train, y_train)
    X_valid, y_valid = OutliersManager.remove_outliers(X_valid, y_valid)

    clf_KNN_lasso(X_train, X_valid, y_train, y_valid)


def DoTesting(season_year, season_type):

    features_data = FeaturesManager.build_features([season_year], season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X_test = features_data[feature_names]
    y_test = features_data[target_name]

    api = API()

    workspace_name = "ift6758-a22-g08"

    # Download and evaluate the Logistic Regresion on Distance
    api.download_registry_model(workspace_name, "KNN_Lasso", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "KNN_Lasso.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    y_proba = clf.predict_proba(X_test)[:,1]
    metrics = evaluate(y_test, y_proba)
    
    print('--------------------------------')
    print('KNN - Lasso')
    print(metrics)



def clf_KNN_lasso(X_train, X_valid, y_train, y_valid):
    experiment = start_experiment()
    
    experiment.set_name('KNN_Lasso')

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

    experiment.log_dataset_hash(X_train)

    # Standard Scaler
    scaler = StandardScaler()

    # Features selection
    # selector = FeaturesSelector.SelectFromLinearSVC()
    selector = SelectFromModel(LinearSVC(
        C=0.01, 
        penalty="l1", 
        dual=False,
        random_state=RANDOM_SEED))


    # Classifier
    n_neighbors = 7
    algorithm = 'auto'
    weights = 'distance'
    clf_KNN = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm)
    
    # Pipeline
    steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ('scaler', scaler), ('selector', selector), ("clf_KNN", clf_KNN)]
    pipeline = Pipeline(steps=steps).fit(X_train, y_train)
    

    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "KNN_Lasso.pkl")
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pipeline, file)
    experiment.log_model("KNN_Lasso", pkl_filename)
    experiment.register_model("KNN_Lasso")

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
        "model_type": "KNN",
        "scaler": scaler,
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Lasso",}
    experiment.log_parameters(params)
    
    experiment.end()
        
if __name__ == "__main__":
    # KNNParameters()
    DoTraining()
    # DoTesting(2019, "Regular")
    # DoTesting(2019, "Playoffs")
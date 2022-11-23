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
    X_valid.drop(labels=['Shot angle', 'Shot distance'], axis=1)

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


def RandomForestHyperParameters(project_name: str):

    # numerical_columns = [
    #        'Shot distance', 'Elapsed time since Power Play', 'Last event elapsed time', 
    #        'st_Y', 'Last event angle', 'Change in Shot Angle', 
    #        'Shot angle']

    # numerical_columns = [
    #     'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
    #     'Speed From Previous Event', 'Change in Shot Angle', 
    #     'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
    #     'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
    #     'Last event distance', 'Last event angle']


    # numerical_columns = [
    #     'Period seconds', 'st_X', 'st_Y', 
    #     'Speed From Previous Event', 'Change in Shot Angle', 
    #     'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
    #     'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
    #     'Last event distance', 'Last event angle']

    # nominal_columns = ['Shot Type', 'Strength', 'Shooter Side', 'Shooter Ice Position', 'Angle Bins', 'Distance Bins']
    # ordinal_columns = ['Period', 'Num players With', 'Num players Against', 'Is Empty', 'Rebound']

    numerical_columns = [
           'Elapsed time since Power Play', 'Last event elapsed time', 
           'st_Y', 'Last event angle', 'Change in Shot Angle']

    nominal_columns = ['Strength', 'Angle Bins', 'Distance Bins']
    ordinal_columns = ['Num players Against', 'Rebound']

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
        "sampling_strategy": {
            "type": "discrete",
            "values": [0.3, 0.4, 0.5, 0.6] },
        # "variance_threshold" : {
        #     "type": "discrete",
        #     "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # },
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
        project_name="best-models",
        workspace="ift6758-a22-g08")

    X_train, X_valid, y_train, y_valid = GetData()

    X_train = X_train[numerical_columns + ordinal_columns + nominal_columns]
    X_valid = X_valid[numerical_columns + ordinal_columns + nominal_columns]

    
    scaler = StandardScaler()

    for experiment in opt.get_experiments():

        n_estimators        = experiment.get_parameter("n_estimators")
        max_depth           = experiment.get_parameter("max_depth")
        sampling_strategy   = experiment.get_parameter("sampling_strategy")
        # variance_threshold  = experiment.get_parameter("variance_threshold")

        # selector = VarianceThreshold(variance_threshold)
        clf_forest = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_SEED)

        # Pipeline
        # steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ('scaler', scaler), ('selector', selector), ("clf_forest", clf_forest)]
        steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ("clf_forest", clf_forest)]
        pipeline = Pipeline(steps=steps)

        run_search(experiment, pipeline, X_train, y_train, cv)
        pipeline.fit(X_train, y_train)

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



def process_data(X, y):

    X, y = OutliersManager.remove_outliers(X, y)
    X['Rebound'] = ((X['Rebound'] == 1) & (X['Last event elapsed time'] < 4)).astype(int)
    
    distance_bins = np.linspace(0,185,10)
    angle_bins = np.linspace(-185,185,10)
    X['Angle Bins'] = pd.cut(X['Shot angle'], bins=angle_bins, include_lowest=True, labels=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8'])
    X['Distance Bins'] = pd.cut(X['Shot distance'], bins=distance_bins, include_lowest=True, labels=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'] )

    X.drop(labels=['Shot angle', 'Shot distance'], axis=1)
    return X, y


def GetTraining():
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X = features_data[feature_names]
    y = features_data[target_name]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    X_train, y_train = process_data(X_train, y_train)
    X_valid, y_valid = process_data(X_valid, y_valid)

    return X_train, X_valid, y_train, y_valid


def GetTesting(season_year, season_type):
    features_data = FeaturesManager.build_features([season_year], season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X_test = features_data[feature_names]
    y_test = features_data[target_name]

    X_test, y_test = process_data(X_test, y_test)

    return X_test, y_test



def DoTraining():

    X_train, X_valid, y_train, y_valid = GetTraining()
    clf_randomforest_binning(X_train, X_valid, y_train, y_valid, RANDOM_SEED)


def DoTesting(season_year, season_type):

    X_test, y_test = GetTesting(season_year, season_type)
    api = API()

    workspace_name = "ift6758-a22-g08"

    # Download and evaluate the Logistic Regresion on Distance
    api.download_registry_model(workspace_name, "RandomForest_Binning", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "RandomForest_Binning.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    y_proba = clf.predict_proba(X_test)[:,1]
    metrics = evaluate(y_test, y_proba)
    
    print('--------------------------------')
    print('RandomForest - Binning')
    print(metrics)



def clf_randomforest_binning(X_train, X_valid, y_train, y_valid, RANDOM_SEED):
    experiment = start_experiment()
    
    experiment.set_name('RandomForest_Binning')

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


    experiment.log_dataset_hash(X_train)

    # Classifier
    n_estimators = 74
    max_depth = 5
    sampling_strategy = 0.4
    clf_forest = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth, 
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_SEED)
    
    # Pipeline
    steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ("clf_forest", clf_forest)]
    pipeline = Pipeline(steps=steps).fit(X_train, y_train)
    

    pkl_filename = './models/Forest_Binning.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pipeline, file)
    experiment.log_model("Forest_Binning", pkl_filename)
    experiment.register_model("Forest_Binning")

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
        "scaler": "StdScaler",
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Binning",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
   

if __name__ == "__main__":
    DoTraining()
    # DoTesting(2019, "Regular")
    # DoTesting(2019, "Playoffs")




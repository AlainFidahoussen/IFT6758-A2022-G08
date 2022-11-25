from dotenv import load_dotenv
load_dotenv();

import src.features.build_features as FeaturesManager
import src.features.select_features as FeaturesSelector
import src.features.detect_outliers as OutliersManager

import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2

from imblearn.pipeline import Pipeline
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from comet_ml import Experiment
from comet_ml import Optimizer
from comet_ml import API

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
    features_data = FeaturesManager.build_features([season_year], season_type)

    features_to_keep = FeaturesManager.GetFeaturesToKeep()
    feature_names, target_name = features_to_keep[0:-1], features_to_keep[-1]
    feature_names = np.array(feature_names)
    
    X_test = features_data[feature_names]
    y_test = features_data[target_name]

    X_test, y_test = OutliersManager.remove_outliers(X_test, y_test)

    return X_test, y_test


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
    
def AdaBoostHyperParameters(project_name: str):

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
            "min": 20,
            "max": 100
        },
        "learning_rate": {
            "type": "discrete",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        "selector_k" : {
            "type": "discrete",
            "values": [8, 9, 10, 11, 12, 13, 14, 15]
        },
        "over_sample" : {
            "type": "discrete",
            "values": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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

    cv = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)

    # initializing the comet ml optimizer
    opt = Optimizer(
        api_key=os.environ.get('COMET_API_KEY'),
        config=config_dict,
        project_name=project_name,
        workspace="ift6758-a22-g08")

    X_train, X_valid, y_train, y_valid = GetTrainingData()
    scaler = StandardScaler()

   
    for experiment in opt.get_experiments():

        n_estimators   = experiment.get_parameter("n_estimators")
        learning_rate  = experiment.get_parameter("learning_rate")
        over_sample    = experiment.get_parameter("over_sample")
        selector_k     = experiment.get_parameter("selector_k")

        selector = SelectKBest(k=selector_k)

        over = RandomOverSampler(sampling_strategy=over_sample)

        clf_adaboost = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=RANDOM_SEED)

        # Pipeline
        steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ('scaler', scaler), ('selector', selector), ("over", over), ("clf_adaboost", clf_adaboost)]
        pipeline = Pipeline(steps=steps)

        run_search(experiment, pipeline, X_train, y_train, cv)

        pipeline.fit(X_train, y_train)
        
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
    X_train, X_valid, y_train, y_valid = GetTrainingData()
    clf_adaboost_anova(X_train, X_valid, y_train, y_valid)


def DoTesting(season_year, season_type):

    X_test, y_test = GetTestingData()

    api = API()

    workspace_name = "ift6758-a22-g08"

    # Download and evaluate the Logistic Regresion on Distance
    api.download_registry_model(workspace_name, "AdaBoost_Anova", "1.0.0", output_path=os.environ["NHL_MODEL_DIR"], expand=True)
    pkl_filename = os.path.join(os.environ["NHL_MODEL_DIR"], "AdaBoost_Anova.pkl")
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    y_proba = clf.predict_proba(X_test)[:,1]
    metrics = evaluate(y_test, y_proba)
    
    print('--------------------------------')
    print('Adaboost - Anova')
    print(metrics)




def clf_adaboost_anova(X_train, X_valid, y_train, y_valid):
    experiment = start_experiment()
    
    experiment.set_name('Adaboost_Anova')

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
    selector_k = 11    
    selector = SelectKBest(k=selector_k)

    # Oversampling
    over_sample = 0.4
    over = RandomOverSampler(sampling_strategy=over_sample)

    # Classifier
    n_estimators = 80
    learning_rate = 0.7
    clf_adaboost = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate, 
        random_state=RANDOM_SEED)
    
    # Pipeline
    steps = [('fill_nan', fill_nan), ('one_hot', one_hot),  ('scaler', scaler), ('selector', selector), ("over", over), ("clf_adaboost", clf_adaboost)]
    pipeline = Pipeline(steps=steps).fit(X_train, y_train)
    

    pkl_filename = './models/AdaBoost_Anova.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pipeline, file)
    experiment.log_model("AdaBoost_Anova", pkl_filename)
    experiment.register_model("AdaBoost_Anova")

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
        "model_type": "adaboost",
        "scaler": scaler,
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Anova",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
   

if __name__ == "__main__":
    DoTraining()
    # DoTesting(2019, "Regular")
    # DoTesting(2019, "Playoffs")


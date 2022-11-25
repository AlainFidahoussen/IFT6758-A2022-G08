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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
import src.features.detect_outliers as OutliersManager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.calibration import CalibratedClassifierCV

from comet_ml import Experiment
from comet_ml import Optimizer
from comet_ml import API

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




def GetTrainingData():
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)
    return features_data


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

class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self._features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._features]

    
class PandasColumnTransformer(ColumnTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)


class PandasKBinsDiscretizer(KBinsDiscretizer):

    def __init__(self, n_bins):
        super(PandasKBinsDiscretizer, self).__init__(n_bins, encode='ordinal')

    def transform(self, X):
        self.col_names = list(X.columns.values)
        X = super(PandasKBinsDiscretizer, self).transform(X)
        X = pd.DataFrame(X, columns=self.col_names)
        return X

    
def DoTraining(calibrated=False):
    training_data = GetTrainingData()
    clf_randomforest_binning_shap_var(training_data, calibrated)


def clf_randomforest_binning_shap_var(training_data, calibrated=False):
    experiment = start_experiment()
    
    experiment.set_name('RandomForest_Binning_SHAP_Var')

    numerical_columns = ['Shot distance', 'Shot angle', 
        'Elapsed time since Power Play', 'Last event elapsed time', 
        'st_Y', 'Last event angle', 'Change in Shot Angle']

    nominal_columns = ['Strength', 'Rebound']
    ordinal_columns = ['Num players Against']

    # median
    fill_nan = ColumnTransformer(transformers = [
        ('cat', SimpleImputer(strategy ='most_frequent'), nominal_columns + ordinal_columns),
        ('num', SimpleImputer(strategy ='median'), numerical_columns),
    ], remainder ='passthrough')

    # one-hot      
    one_hot = ColumnTransformer(transformers = [
        ('enc', OneHotEncoder(sparse = False), list(range(len(nominal_columns)))),
    ], remainder ='passthrough')

    # Binning on Distance and angles
    binner_on_numeric = PandasColumnTransformer(transformers=[
                ("binner",  PandasKBinsDiscretizer(10),['Shot distance', 'Shot angle'])
    ], remainder='passthrough')

    # columns selector
    columns_selector = ColumnsSelector(numerical_columns + nominal_columns + ordinal_columns)

    experiment.log_dataset_hash(training_data)

    X, y = training_data.drop(labels=['Is Goal'], axis=1), training_data['Is Goal']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
  
    X_train, y_train = OutliersManager.remove_outliers(X_train, y_train)

    # Classifier
    max_depth = 10	
    n_estimators = 170
    sampling_strategy = 0.4	
    clf_forest = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth, 
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_SEED)
    
    variance_threshold = 0.4
    selector = VarianceThreshold(variance_threshold)

    scaler = StandardScaler()

    # Pipeline
    if calibrated:
        steps = [("columns_selector", columns_selector), 
                 ("binning", binner_on_numeric), 
                 ('fill_nan', fill_nan), 
                 ('one_hot', one_hot), 
                 ('scaler', scaler), 
                 ('selector', selector), 
                 ("clf_forest",  CalibratedClassifierCV(base_estimator=clf_forest, cv=3))]
    else:
        steps = [("columns_selector", columns_selector), 
                 ("binning", binner_on_numeric), 
                 ('fill_nan', fill_nan), 
                 ('one_hot', one_hot), 
                 ('scaler', scaler), 
                 ('selector', selector), 
                 ("clf_forest", clf_forest)]
        
    pipeline = Pipeline(steps=steps).fit(X_train, y_train)
    
    if calibrated:
        filename = 'Forest_Binning_SHAP_Var_calibrated'
    else:
        filename = 'Forest_Binning_SHAP_Var'
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
        "scaler": scaler,
        # "param_grid":str(param_grid),
        "stratify":True, 
        "data": "Binning",}
    experiment.log_parameters(params)
    
    experiment.end()
    
    
   

if __name__ == "__main__":
    DoTraining(calibrated=True)
    # DoTesting(2019, "Regular")
    # DoTesting(2019, "Playoffs")



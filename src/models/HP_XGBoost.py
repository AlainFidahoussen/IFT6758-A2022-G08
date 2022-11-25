import os
import sys
import numpy as np
import seaborn as sns

# To load the environment variable defined in the .env file
from dotenv import load_dotenv
load_dotenv();

import src.visualization.visualize as VizManager

# import comet_ml at the top of your file
from comet_ml import Experiment
from comet_ml import Optimizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from xgboost import XGBClassifier

RANDOM_SEED = 42

def evaluate(y_true, y_proba):
    y_pred = np.round(y_proba)
    return {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'macro f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'rocauc': roc_auc_score(y_true, y_proba)
    }

def XGBoost_GridSearch(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = XGBClassifier()
    grid_search = GridSearchCV(
        XGBClassifier(),
        param_grid={'learning_rate': [0.01, 0.1, 0.2, 0.3], 'gamma': [0, 2, 4, 6], 'max_depth': [4, 6, 8]},
        scoring={"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)},
        refit="AUC",
        # verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    api_key = os.environ.get('COMET_API_KEY')

    # source: Comet.ml scikit-learn documentation
    for i in range(len(grid_search.cv_results_['params'])):
        experiment = Experiment(
            api_key=api_key,
            project_name="hyperparameters-xgboost",
            workspace="ift6758-a22-g08",
        )
        for k,v in grid_search.cv_results_.items():
            if k == "params":
                experiment.log_parameters(v[i])
            else:
                experiment.log_metric(k,v[i])

    return grid_search, X_test
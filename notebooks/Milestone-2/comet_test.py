# Demonstration of how to use comet.ml 
# Does not seem to work with .ipynb file? Kernel Extremely slow...


# To load the environment variable defined in the .env file
from dotenv import load_dotenv
load_dotenv();

# import comet_ml at the top of your file
from comet_ml import Experiment

import sys
import os

# Only for me
script_dir = os.path.dirname( __file__ )
module_dir = os.path.join( script_dir, '../..')
sys.path.append(module_dir)

# Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file with containing the api key
    project_name="milestone-2",
    workspace="ift6758-a22-g08",
)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick


from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

import src.visualization.visualize as VizManager
import src.features.build_features as FeaturesManager

SEED = 42

seasons_year = [2015, 2016, 2017, 2018]
season_type = "Regular"
features_data = FeaturesManager.build_features(seasons_year, season_type)

# We take the absolute value, for symmetry reasons
features_data['Shot angle'] = features_data['Shot angle'].abs()

distance_data = features_data[['Shot distance', 'Is Goal']].dropna()
X = distance_data[['Shot distance']]
y = distance_data['Is Goal']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

clf = LogisticRegression(random_state=SEED).fit(X_train, y_train)

from sklearn.metrics import accuracy_score

def evaluate(y_true, y_pred):
    return {
      'accuracy': accuracy_score(y_true, y_pred)
    }

# Log Training Metrics
y_train_pred = clf.predict(X_train)

with experiment.train():
    metrics = evaluate(y_train, y_train_pred)
    experiment.log_metrics(metrics)

# Log Test Metrics
y_valid_pred = clf.predict(X_valid)

with experiment.test():
    metrics = evaluate(y_valid, y_valid_pred)
    experiment.log_metrics(metrics)
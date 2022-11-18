# To load the environment variable defined in the .env file
from dotenv import load_dotenv
load_dotenv();

import os

# import comet_ml at the top of your file
from comet_ml import Experiment
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import src.visualization.visualize as VizManager
import src.features.build_features as FeaturesManager
from sklearn.metrics import classification_report

from joblib import dump, load


# Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'), # create a COMET_API_KEY env var in the .env file with containing the api key
    project_name="milestone-2",
    workspace="ift6758-a22-g08",
)

def evaluate(y_true, y_pred):
    return {
      'f1-score': f1_score(y_true, y_pred)
    }

RANDOM_SEED = 42

def Do():
    seasons_year = [2015, 2016, 2017, 2018]
    season_type = "Regular"
    features_data = FeaturesManager.build_features(seasons_year, season_type)

    # We take the absolute value, for symmetry reasons
    features_data['Shot angle'] = features_data['Shot angle'].abs()

    X = features_data[['Shot distance', 'Shot angle']]
    y = features_data['Is Goal']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # Logistic Regression - Distance only
    X_distance_train, X_distance_valid = X_train[['Shot distance']], X_valid[['Shot distance']]
    clf_distance = LogisticRegression(random_state=RANDOM_SEED).fit(X_distance_train, y_train)
    y_pred = clf_distance.predict(X_distance_valid)
    report_distance = classification_report(y_valid, y_pred, output_dict=True)
    print(report_distance)
    filename_model = './models/LogisticRegression_Distance.joblib'
    dump(clf_distance, filename_model) 
    experiment.log_model("Logistic Regression", filename_model)
    # experiment.register_model("Logistic Regression")

    # experiment.log_dataset_hash(X_distance_train)
    experiment.log_metrics(report_distance['macro avg'], prefix='macro avg')
    # experiment.log_confusion_matrix(y_valid, y_pred)


    # Logistic Regression - Angle only
    X_angle_train, X_angle_valid = X_train[['Shot angle']], X_valid[['Shot angle']]
    clf_angle = LogisticRegression(random_state=RANDOM_SEED).fit(X_angle_train, y_train)

    # # Log Metrics
    # with experiment.train():
    #     y_train_pred = clf_angle.predict(X_angle_train)
    #     metrics = evaluate(y_train, y_train_pred)
    #     experiment.log_metrics(metrics)

    # with experiment.test():
    #     y_valid_pred = clf_angle.predict(X_angle_valid)
    #     metrics = evaluate(y_valid, y_valid_pred)
    #     experiment.log_metrics(metrics)



    # Logistic Regression - Distance and Angle
    X_distance_angle_train, X_distance_angle_valid = X_train, X_valid
    clf_distance_angle = LogisticRegression(random_state=RANDOM_SEED).fit(X_distance_angle_train, y_train)

    # # Log Metrics
    # with experiment.train():
    #     y_train_pred = clf_distance_angle.predict(X_distance_angle_train)
    #     metrics = evaluate(y_train, y_train_pred)
    #     experiment.log_metrics(metrics)

    # with experiment.test():
    #     y_valid_pred = clf_distance_angle.predict(X_distance_angle_valid)
    #     metrics = evaluate(y_valid, y_valid_pred)
    #     experiment.log_metrics(metrics)

    # experiment.end()

if __name__ == "__main__":
    Do()

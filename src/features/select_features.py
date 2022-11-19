from dotenv import load_dotenv
load_dotenv();

RANDOM_SEED = 42

import os

from sklearn.ensemble import RandomForestClassifier

import pickle


from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.base import BaseEstimator, TransformerMixin


class SelectFromTree(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        dir = os.path.join(os.environ['NHL_MODEL_DIR'], 'FeaturesSelector')
        self.pkl_dir = dir

    def fit(self, X, y=None):

        filename = os.path.join(self.pkl_dir, 'forest.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                selector = pickle.load(file)
            return selector

        else:
            selector = SelectFromModel(RandomForestClassifier(
                random_state=RANDOM_SEED, 
                n_estimators = 100, 
                class_weight='balanced'))
            selector.fit(X, y)

            os.makedirs(self.pkl_dir, exist_ok=True)
            with open(os.path.join(self.pkl_dir, 'forest.pkl'), 'wb') as file:
                pickle.dump(selector, file)

        return self


    def transform(self, X, y=None):
        
        filename = os.path.join(self.pkl_dir, 'forest.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                selector = pickle.load(file)
        
            return selector.transform(X), y
        else:
            return X, y


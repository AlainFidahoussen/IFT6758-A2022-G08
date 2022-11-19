from dotenv import load_dotenv
load_dotenv();

RANDOM_SEED = 42

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

from sklearn.base import BaseEstimator, TransformerMixin


class SelectFromRandomForest(BaseEstimator, TransformerMixin):

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




class SelectFromTree_RecursiveElimination(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        dir = os.path.join(os.environ['NHL_MODEL_DIR'], 'FeaturesSelector')
        self.pkl_dir = dir

    def fit(self, X, y=None):

        filename = os.path.join(self.pkl_dir, 'tree_recursive_elimination.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                selector = pickle.load(file)
            return selector

        else:

            min_features_to_select = 1

            selector = RFECV(
                estimator=DecisionTreeClassifier(),
                step=1,
                cv=StratifiedKFold(2),
                scoring="f1_macro",
                min_features_to_select=min_features_to_select,
            )
            selector.fit(X, y)


            os.makedirs(self.pkl_dir, exist_ok=True)
            with open(os.path.join(self.pkl_dir, 'tree_recursive_elimination.pkl'), 'wb') as file:
                pickle.dump(selector, file)

        return self


    def transform(self, X, y=None):
        
        filename = os.path.join(self.pkl_dir, 'tree_recursive_elimination.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                selector = pickle.load(file)
        
            return selector.transform(X), y
        else:
            return X, y


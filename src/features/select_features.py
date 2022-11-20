from dotenv import load_dotenv
load_dotenv();

RANDOM_SEED = 42

import os

from sklearn.ensemble import RandomForestClassifier

import pickle


from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


class SelectFromRandomForest(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        self.selector = None

    def fit(self, X, y=None):

        self.selector = SelectFromModel(RandomForestClassifier(
                random_state=RANDOM_SEED, 
                n_estimators = 50, 
                class_weight='balanced'))
        self.selector.fit(X, y)

        return self.selector

    def transform(self, X, y=None):
        if self.selector is not None: 
            X_new = self.selector.transform(X)
            return X_new
        else:
            return X


class SelectFromTree_RecursiveElimination(BaseEstimator, TransformerMixin):

    def init(self, **kwargs):
        self.selector = None

    def fit(self, X, y=None):

        min_features_to_select = 1

        self.selector = RFECV(
            estimator=DecisionTreeClassifier(),
            step=1,
            cv=StratifiedKFold(2),
            scoring="f1_macro",
            min_features_to_select=min_features_to_select)
        self.selector.fit(X, y)

        return self.selector


    def transform(self, X, y=None):
        if self.selector is not None: 
            X_new = self.selector.transform(X)
            return X_new
        else:
            return X


        
class SelectFromLinearSVC_II(BaseEstimator, TransformerMixin):

    def init(self, **kwargs):
        self.selector = None


    def fit(self, X, y=None):

        self.selector = SelectFromModel(LinearSVC(
                C=0.01, 
                penalty="l1", 
                dual=False,
                random_state=RANDOM_SEED))
        self.selector.fit(X, y)

        return self.selector

    def transform(self, X, y=None):
        if self.selector is not None: 
            X_new = self.selector.transform(X)
            return X_new
        else:
            return X

        
        
class SelectFromLinearSVC(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        dir = os.path.join(os.environ['NHL_MODEL_DIR'], 'FeaturesSelector')
        self.pkl_dir = dir
        self.selected_features = None

    def fit(self, X, y=None):

        filename = os.path.join(self.pkl_dir, 'linearSVC.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                selector = pickle.load(file)
            # feature_names = np.array(X.columns)
            # self.selected_features = feature_names[selector.get_support()]
            
            return selector

        else:
            selector = SelectFromModel(LinearSVC(
                C=0.01, 
                penalty="l1", 
                dual=False,
                random_state=RANDOM_SEED))
            # selector.fit(X, y)
            # feature_names = np.array(X.columns)
            # self.selected_features = feature_names[selector.get_support()]

            os.makedirs(self.pkl_dir, exist_ok=True)
            with open(os.path.join(self.pkl_dir, 'linearSVC.pkl'), 'wb') as file:
                pickle.dump(selector, file)

        return self


    def transform(self, X, y=None):
        
        filename = os.path.join(self.pkl_dir, 'linearSVC.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                selector = pickle.load(file)
        
            # return pd.DataFrame(selector.transform(X), columns=self.selected_features), y
            return selector.transform(X), y
        else:
            return X, y

        
class SelectFromPCA():

    def __init__(self, n_components):
        self.n_components = n_components
        self.selector = None
        
    
    def separate_X(self, X) :
        continuous_col = ['Period seconds', 'st_X', 'st_Y', 'Shot distance',
           'Shot angle', 'Speed From Previous Event',
           'Change in Shot Angle', 'Shooter Goal Ratio Last Season',
           'Goalie Goal Ratio Last Season', 'Elapsed time since Power Play',
           'Last event elapsed time', 'Last event st_X', 'Last event st_Y',
           'Last event distance', 'Last event angle']

        X_cont, X_other = X[continuous_col], X.drop(continuous_col, axis=1)
        X_other.reset_index(drop=True, inplace=True)
        
        X_cont_st = StandardScaler().fit_transform(X_cont)
        
        return X_cont_st, X_other
        
    def fit(self, X, y=None):
        
        X_cont_st, X_other = self.separate_X(X)
        
        self.selector = PCA(n_components=self.n_components)
        self.selector.fit(X_cont_st)  

        return self.selector


    def transform(self, X, y=None):
        if self.selector is not None: 
            
            X_cont_st, X_other = self.separate_X(X)
    
            X_PCA = self.selector.transform(X_cont_st)
            
            principalDf = pd.DataFrame(data = X_PCA, columns = [f'PC{i}' for i in range(1, self.n_components+1)])
            X_final = pd.concat([principalDf, X_other], axis = 1)
            
            cev = np.cumsum(self.selector.explained_variance_ratio_)
            cev = np.insert(cev, 0, 0)
            print(f'Cumulative explained variance with {self.n_components} components: {cev[-1]}')

            # plt.figure(figsize=(15,10))
            # plt.ylim(0.0,1.1)
            # plt.plot(cev, linewidth=3)
            # plt.xlabel('number of components', fontsize=21)
            # plt.ylabel('cumulative explained variance', fontsize=21)
            # plt.title('Scree Plot using PCA', fontsize=24)
            # plt.rc('font', size=16)
            # plt.grid()
            # plt.show()
        
            return X_final, y
        else:
            return X, y
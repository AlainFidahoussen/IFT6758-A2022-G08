from dotenv import load_dotenv
load_dotenv();

RANDOM_SEED = 42

import os

from sklearn.ensemble import RandomForestClassifier

import pickle


#from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC, SVC
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

        
class SelectFromLinearSVC(BaseEstimator, TransformerMixin):

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
        
        col_names = list(X.columns)
        
        X_st = StandardScaler().fit_transform(X)
        X_st = pd.DataFrame(X_st, columns=col_names)
        
        X_cont_st, X_other_st = X_st[continuous_col], X_st.drop(continuous_col, axis=1)
        
        return X_cont_st, X_other_st
        
    def fit(self, X, y=None):
        
        X_cont_st, X_other_st = self.separate_X(X)
        
        self.selector = PCA(n_components=self.n_components)
        self.selector.fit(X_cont_st)  

        return self.selector


    def transform(self, X, y=None):
        if self.selector is not None: 
            
            X_cont_st, X_other_st = self.separate_X(X)
    
            X_PCA = self.selector.transform(X_cont_st)
            
            # principalDf = pd.DataFrame(data = X_PCA, columns = [f'PC{i}' for i in range(1, self.n_components+1)])
            # X_final = pd.concat([principalDf, X_other], axis = 1)
            X_final = np.hstack((X_PCA, X_other_st))
            
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

class SelectFromVarianceThreshold(BaseEstimator, TransformerMixin):

    def __init__(self,threshold=0.5):
        self.selector = None
        self.threshold = threshold

    def fit(self, X, y=None):

        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X, y)
        return self.selector


    def transform(self, X, y=None):
        if self.selector is not None: 
            X_new = self.selector.transform(X)
            return X_new,y
        else:
            return X,y
        
class SelectFromKBest_chi2(BaseEstimator, TransformerMixin):

    def __init__(self,k=22):
        self.selector = None
        self.k = k
        
    def separate_X(self, X) :
        numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']
        
        categorical_columns = [c for c in X.columns if c not in numerical_columns]

        X_cat, X_num =  X[categorical_columns], X[numerical_columns].to_numpy()
        
        return X_cat, X_num

    def fit(self, X, y=None):
        
        X_cat, X_num = self.separate_X(X)
        self.selector = SelectKBest(score_func=chi2, k=self.k)
        self.selector.fit(X_cat, y)
        return self.selector


    def transform(self, X, y=None):
        X_cat, X_num = self.separate_X(X)
        if self.selector is not None: 
            X_new = self.selector.transform(X_cat)
            X_final = np.hstack((X_new, X_num))
            return X_final,y
        else:
            return X,y

class SelectFromKBest_MI(BaseEstimator, TransformerMixin):

    def __init__(self,k=22):
        self.selector = None
        self.k = k
        
    def separate_X(self, X) :
        numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']
        
        categorical_columns = [c for c in X.columns if c not in numerical_columns]

        X_cat, X_num =  X[categorical_columns], X[numerical_columns].to_numpy()
        
        return X_cat, X_num

    def fit(self, X, y=None):
        
        X_cat, X_num = self.separate_X(X)
        self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        self.selector.fit(X_cat, y)

        return self.selector


    def transform(self, X, y=None):
        X_cat, X_num = self.separate_X(X)
        if self.selector is not None: 
            X_new = self.selector.transform(X_cat)
            X_final = np.hstack((X_new, X_num))
            return X_final,y
        else:
            return X,y
        
class SelectFromAnova(BaseEstimator, TransformerMixin):

    def __init__(self,k=15):
        self.selector = None
        self.k = k
        
    def separate_X(self, X) :
        numerical_columns = [
        'Period seconds', 'st_X', 'st_Y', 'Shot distance', 'Shot angle', 
        'Speed From Previous Event', 'Change in Shot Angle', 
        'Shooter Goal Ratio Last Season', 'Goalie Goal Ratio Last Season',
        'Elapsed time since Power Play', 'Last event elapsed time', 'Last event st_X', 'Last event st_Y', 
        'Last event distance', 'Last event angle']
        
        categorical_columns = [c for c in X.columns if c not in numerical_columns]

        X_num, X_cat =  X[numerical_columns], X[categorical_columns].to_numpy()
        
        return X_num, X_cat

    def fit(self, X, y=None):
        
        X_num, X_cat = self.separate_X(X)
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selector.fit(X_num, y)

        return self.selector


    def transform(self, X, y=None):
        X_num, X_cat = self.separate_X(X)
        if self.selector is not None: 
            X_new = self.selector.transform(X_num)
            X_final = np.hstack((X_new, X_cat))
            return X_final,y
        else:
            return X,y

        
class SelectFromSVCForward(BaseEstimator, TransformerMixin):

    def __init__(self, n = 37):
        self.selector = None
        self.n = n
        

    def fit(self, X, y=None):

        self.selector = SequentialFeatureSelector(SVC(), scoring="f1_macro",cv=StratifiedKFold(2),n_features_to_select=self.n, direction="forward")
        self.selector.fit(X, y)
        return self.selector

    def transform(self, X, y=None):
        if self.selector is not None: 
            X_new = self.selector.transform(X)
            return X_new
        else:
            return X
        
class SelectFromSVCBackward(BaseEstimator, TransformerMixin):

    def __init__(self, n = 37):
        self.selector = None
        self.n = n

    def fit(self, X, y=None):

        self.selector = SequentialFeatureSelector(SVC(), scoring="f1_macro",cv=StratifiedKFold(2),n_features_to_select=self.n, direction="backward")
        self.selector.fit(X, y)
        return self.selector

    def transform(self, X, y=None):
        if self.selector is not None: 
            X_new = self.selector.transform(X)
            return X_new,y
        else:
            return X,y


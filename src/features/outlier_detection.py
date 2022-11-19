import pandas as pd
import numpy as np
from scipy import stats

# Removes all the rows of selected feature outliers

def Percentile(X: pd.DataFrame, y: pd.Series, feature: str ='st_X', threshold : list =[0.05, 0.95]) -> (pd.DataFrame, pd.Series): 
    min_thresold, max_thresold = X[feature].quantile(threshold) # define threshold

    percentile_mask = ~(X[feature]<min_thresold) | (X[feature]>max_thresold) # create bool mask for threshold

    X_percentile = X[percentile_mask]
    y_percentile  = y[percentile_mask]

    return X_percentile, y_percentile


def IQR(X: pd.DataFrame, y: pd.Series, feature: str ='st_X') -> (pd.DataFrame, pd.Series):
    Q1 = X[feature].quantile(0.25)
    Q3 = X[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR

    IQR_mask = ~(X[feature]<lower_limit) | (X[feature]>upper_limit)
    X_IQR = X[IQR_mask]
    y_IQR = y[IQR_mask]

    return X_IQR, y_IQR


def STD(X: pd.DataFrame, y: pd.Series, feature: str ='st_X', factor: int =2) -> (pd.DataFrame, pd.Series):
    upper_limit = X[feature].mean() + factor*X[feature].std()
    lower_limit = X[feature].mean() - factor*X[feature].std()

    std_mask = (X[feature]<upper_limit) & (X[feature]>lower_limit)

    X_std = X[std_mask]
    y_std = y[std_mask]

    return X_std, y_std


def zscore(X: pd.DataFrame, y: pd.Series, feature: str ='st_X', threshold: int =3) -> (pd.DataFrame, pd.Series):
    zscore_mask = ~(np.abs(stats.zscore(X[feature])) > threshold)

    X_zscore = X[zscore_mask]
    y_zscore = y[zscore_mask]

    return X_zscore, y_zscore


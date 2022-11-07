import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter 

from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

# To use the same colors in the multiple plots of Milestone2
colors = ['red', 'blue', 'green', 'yellow', 'orange']

def get_shots_coordinates(data_season_df:pd.DataFrame, team:str=None):

    # Return the mean for all the team
    if team is None:
        x_shots = data_season_df['st_X'].to_numpy().copy().astype(np.float32)
        y_shots = data_season_df['st_Y'].to_numpy().copy().astype(np.float32)

        x_nan = np.isnan(x_shots)
        y_nan = np.isnan(y_shots)

        x_shots = x_shots[(~x_nan) & (~y_nan)]
        y_shots = y_shots[(~x_nan) & (~y_nan)]

    # Return the mean just for the team
    else:
        data_one_team_df = data_season_df.loc[data_season_df['Team'] == team]
        if data_one_team_df.size == 0:
            return [[],[]]
        
        x_shots = data_one_team_df['st_X'].to_numpy().copy().astype(np.float32)
        y_shots = data_one_team_df['st_Y'].to_numpy().copy().astype(np.float32)

    # Remove the nan value for both x and y
    x_nan = np.isnan(x_shots)
    y_nan = np.isnan(y_shots)
    x_shots = x_shots[(~x_nan) & (~y_nan)]
    y_shots = y_shots[(~x_nan) & (~y_nan)]

    return [x_shots, y_shots]


def get_shots_hist2D(x_shots, y_shots, num_pts_x:int=40, num_pts_y:int=20):

    # We are only interested in shots in offensive zone, so we don't care about negative x coordinates
    x_min, x_max = 0.0, 100.
    y_min, y_max = -42.5, 42.5

    delta_x = (x_max-x_min) / num_pts_x
    delta_y = (y_max-y_min) / num_pts_y

    x_grid = np.arange(x_min-delta_x, x_max+delta_x, delta_x)
    y_grid = np.arange(y_min-delta_y, y_max+delta_y, delta_y)

    H, x_edge, y_edge = np.histogram2d(x_shots, y_shots, bins=[x_grid, y_grid])
    
    return H.T, x_edge[1:], y_edge[1:]


def compute_diff_shots(data_season_df:pd.DataFrame, num_pts_x:int=40, num_pts_y:int=20) -> dict:

    dict_diff = {}

    [x_shots_season, y_shots_season] = get_shots_coordinates(data_season_df)
    
    shots_hist2D_season, x_grid, y_grid = get_shots_hist2D(x_shots_season, y_shots_season, num_pts_x=num_pts_x, num_pts_y=num_pts_y)
    number_of_games_season = len(data_season_df['Game ID'].unique())
    shots_hist2D_season_by_hour = shots_hist2D_season / (number_of_games_season*2)

    teams = np.sort(data_season_df['Team'].unique())
    df_number_of_games_by_team = data_season_df[['Team', 'Game ID']].groupby('Team').describe()['Game ID']['unique']

    for team in teams:

        [x_shots_one_team, y_shots_one_team] = get_shots_coordinates(data_season_df, team)
    

        shots_hist2D_one_team, x_grid, y_grid = get_shots_hist2D(x_shots_one_team, y_shots_one_team, num_pts_x=num_pts_x, num_pts_y=num_pts_y)
        shots_hist2D_one_team_by_hour = shots_hist2D_one_team / df_number_of_games_by_team[team]

        diff = gaussian_filter(shots_hist2D_one_team_by_hour-shots_hist2D_season_by_hour, 1.)

        # Normalize between -1 and 1
        diff_min = diff.min()
        diff_max = diff.max()
        alpha = (-2./(diff_min-diff_max)) 
        beta = (diff_min + diff_max) / (diff_min - diff_max)
        diff_norm = alpha * diff + beta

        # Remove shots behind the goals
        mask = np.where(x_grid > 89)
        diff_norm[:, mask] = None
        dict_diff[team] = diff_norm
        

    return dict_diff, x_grid, y_grid


def plot_ROC(classifiers_tuple: list[tuple], add_random=True) -> None:

    plt.figure(figsize=(8, 8))

    for count, classifier in enumerate(classifiers_tuple):

        clf = classifier[0]
        clf_name = classifier[1]
        X = classifier[2]
        y = classifier[3]

        y_pred = clf.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y.ravel(), y_pred.ravel())
        roc_auc = auc(fpr, tpr)


        plt.plot(fpr, tpr, color=colors[count], label=f"{clf_name}: AUC = %0.2f" % roc_auc)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")

    if add_random:
        plt.plot([0, 1], [0, 1], color="black", label='Random Uniform (AUC = 0.5)', linestyle="--")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.show()



def plot_Cumulative_Goal(classifiers_tuple: list[tuple], add_random=True) -> None:

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for count, classifier in enumerate(classifiers_tuple):

        clf = classifier[0]
        clf_name = classifier[1]
        X = classifier[2]
        y = classifier[3]

        y_pred = clf.predict_proba(X)[:,1]
        
        x = np.linspace(0, 100, 20)
        percentiles = [np.percentile(y_pred, i) for i in x]

        goals_ratio = []
        num_total_goals = (y == 1).sum()

        for count_p, _ in enumerate(x[:-1]):

            ind = (y_pred >= percentiles[count_p]) & (y_pred < percentiles[count_p+1])
            num_goals = (y[ind] == 1).sum()
            ratio = num_goals / num_total_goals
            goals_ratio.append(100.*ratio)

        goals_ratio = 100-np.cumsum(np.array(goals_ratio))
        plt.plot(x[0:-1], goals_ratio, color=colors[count], label=clf_name)


    if add_random:
        goals_ratio = []
        y_random = np.random.uniform(low=0.0, high=1.0, size=len(y))
        percentiles = [np.percentile(y_random, i) for i in x]
        for count_p, _ in enumerate(x[:-1]):

            ind = (y_random >= percentiles[count_p]) & (y_random < percentiles[count_p+1])
            num_goals = (y[ind] == 1).sum()
            ratio = num_goals / num_total_goals
            goals_ratio.append(100.*ratio)

        goals_ratio = 100-np.cumsum(np.array(goals_ratio))
        plt.plot(x[0:-1], goals_ratio, color='black', label='Random Uniform')

    plt.title('Cumulative % of goals')
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Proportion')
    plt.xlim([100.0, 0.0])
    plt.ylim([0.0, 100.0])
    plt.yticks(range(0, 110, 10));
    plt.legend(loc="upper left");
    plt.show()


def plot_Goal_Rate(classifiers_tuple: list[tuple], add_random=True) -> None:

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for count, classifier in enumerate(classifiers_tuple):

        clf = classifier[0]
        clf_name = classifier[1]
        X = classifier[2]
        y = classifier[3]

        y_pred = clf.predict_proba(X)[:,1]

        x = np.linspace(0, 100, 20)
        percentiles = [np.percentile(y_pred, i) for i in x]

        goals_over_total = []


        for count_p, _ in enumerate(x[:-1]):

            ind = (y_pred >= percentiles[count_p]) & (y_pred < percentiles[count_p+1])
            num_goals = (y[ind] == 1).sum()
            num_no_goals = (y[ind] == 0).sum()
            ratio = num_goals / (num_goals + num_no_goals)
            goals_over_total.append(100.*ratio)

        goals_over_total = np.array(goals_over_total)

        plt.plot(x[0:-1], goals_over_total, color=colors[count], label=clf_name)


    if add_random:
        goals_over_total = []
        percentiles = [np.percentile(y_pred, i) for i in x]
        y_random = np.random.uniform(low=0.0, high=1.0, size=len(y))
        for count_p, _ in enumerate(x[:-1]):

            ind = (y_random >= percentiles[count_p]) & (y_random < percentiles[count_p+1])
            num_goals = (y[ind] == 1).sum()
            num_no_goals = (y[ind] == 0).sum()
            ratio = num_goals / (num_goals + num_no_goals)
            goals_over_total.append(100.*ratio)

        goals_over_total = np.array(goals_over_total)
        plt.plot(x[0:-1], goals_over_total, color='black', label='Random Uniform')

    plt.title('Goal Rate')
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Goals / (Shots + Goals)')
    plt.xlim([100.0, 0.0])
    plt.ylim([0.0, 100.0])
    plt.yticks(range(0, 110, 10))

    plt.legend(loc="upper left");
    plt.show()


def plot_Calibration(classifiers_tuple: list[tuple], add_random=True) -> None:
    calibration_displays = {}
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    nbins = 50

    for count, classifier in enumerate(classifiers_tuple):

        clf = classifier[0]
        clf_name = classifier[1]
        X = classifier[2]
        y = classifier[3]
        y_pred = clf.predict_proba(X)[:,1]
        
        display = CalibrationDisplay.from_predictions(y, y_pred, n_bins=nbins, name=clf_name, ax=ax_calibration_curve, color=colors[count])
        calibration_displays[clf_name] = display

    if add_random:
        y_random = np.random.uniform(low=0.0, high=1.0, size=len(y))
        CalibrationDisplay.from_predictions(y, y_random, n_bins=nbins, name='Random Uniform', ax=ax_calibration_curve, color='black')

    ax_calibration_curve.set_title("Calibration plots");
    plt.legend(loc="upper left")
    plt.show()


def plot_all_figures(classifiers_tuple: list[tuple], add_random=True) -> None:

    # ----------------------------------------------
    # First plot: ROC Curve
    plot_ROC(classifiers_tuple, add_random)

    # ----------------------------------------------
    # Second plot: Goal Rate
    plot_Goal_Rate(classifiers_tuple, add_random)

    # ----------------------------------------------
    # Third plot: Cumulative % of goals
    plot_Cumulative_Goal(classifiers_tuple, add_random)

    # ----------------------------------------------
    # Last plot : calibration 
    plot_Calibration(classifiers_tuple, add_random)
import HP_AdaBoost
import HP_EasyEnsemble
import HP_GradientBoost
import HP_RandomForest

if __name__ == "__main__":

    # HP_EasyEnsemble.EasyEnsembleParameters("Hyperparameters-EasyEnsemble_PCASelector")
    HP_RandomForest.RandomForestHyperParameters("Hyperparameters-BalancedRandomForest_VarianceThresholdSelector")
    # HP_GradientBoost.GradientBoostHyperParameters("Hyperparameters-GradientBoost_RandomForestSelector")
    # HP_AdaBoost.AdaBoostHyperParameters("Hyperparameters-Adaboost_AnovaSelector")

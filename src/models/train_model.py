import HP_AdaBoost
import HP_EasyEnsemble
import HP_GradientBoost
import HP_RandomForest
import HP_XGBoost_2

if __name__ == "__main__":

    # HP_EasyEnsemble.EasyEnsembleParameters("Hyperparameters-EasyEnsemble_PCASelector")
    # HP_RandomForest.RandomForestHyperParameters("Hyperparameters-RandomForest_VarianceThresholdSelector")
    # HP_GradientBoost.GradientBoostHyperParameters("Hyperparameters-GradientBoost_RandomForestSelector")
    # HP_AdaBoost.AdaBoostHyperParameters("Hyperparameters-Adaboost_AnovaSelector")
    # HP_XGBoost_2.XGBoostHyperParameters("Hyperparameters-Test")
    # HP_RandomForest.RandomForestHyperParameters("Hyperparameters-RandomForest_SHAP")

    # HP_RandomForest.RandomForestHyperParameters("Hyperparameters-RandomForest_Binning")

    # HP_RandomForest.RandomForestHyperParameters("Hyperparameters-RandomForest_SHAP_Binning_VarianceThreshold")
    # HP_AdaBoost.AdaBoostHyperParameters("Hyperparameters-Adaboost_AnovaSelector")
    # HP_GradientBoost.GradientBoostHyperParameters("Hyperparameters-GradientBoost_SHAP_Binning")
    # HP_XGBoost_2.XGBoostHyperParameters("Hyperparameters-XGBoost_VarianceThreshold_Binning")

    HP_RandomForest.RandomForestHyperParameters("Hyperparameters-RandomForest")

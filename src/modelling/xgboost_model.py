# FOR INTERACTIVE SESSION comment out later
import os
# os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")
os.chdir(r"/home/lv/Documents/projects/afl-win-predictor")

# Libraries
import optuna
import mlflow
import datetime
import mlflow.xgboost
import numpy as np
import polars as pl
import xgboost as xgb

from sklearn.metrics import accuracy_score, roc_auc_score
from src.config import PROJECT_ROOT
from src.features.fixture_features import compute_fixture_features
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_training_history,
    plot_feature_importance,
    plot_train_val_test_auc,
    plot_calibration_curve,
    plot_precision_recall,
    plot_accuracy_vs_threshold
)

#####################
# Functions
#####################


#####################
# Code
#####################
from src.modelling.optuna_search import prepare_features

path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
df = compute_fixture_features(path)

# prepare features
cat_cols = ["ateam", "hteam", "venue_location", "time_of_day"]
df_model = prepare_features(df, cat_cols)

############
# mlfow 
############
# TODO: create optuna plots in mlflow
# load the best performing model from mlflow

# Create the study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300, n_jobs=-1)  # n_trials can be larger

# Best trial
best_trial = study.best_trial
print("Best Mean Validation Accuracy:", best_trial.value)
print("Best Hyperparameters:", best_trial.params)

# get the best avg threshold
best_avg_threshold = get_best_mlflow_avg_threshold(study, "20260317_190808")
print("Best Avg Threshold:", best_avg_threshold)

##########################################
# RETRAIN THE BEST PERFORMING MODEL
##########################################
# Separate test season
test_df = df_model.filter(pl.col("year") == 2025)

# Combine all historical seasons for training
train_df = df_model.filter(pl.col("year") < 2025)

# Features and labels
X_train = train_df.drop(["win", "weight", "year"])
y_train = train_df["win"]
w_train = train_df["weight"]

X_test = test_df.drop(["win", "weight", "year"])
y_test = test_df["win"]
w_test = test_df["weight"]

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test, enable_categorical=True)

# Compute scale_pos_weight from all training data (excluding dummy rows if needed)
y_train_np = train_df.filter(pl.col("weight") == 1.0)["win"].to_numpy()
pos = (y_train_np == 1).sum()
neg = (y_train_np == 0).sum()
scale_pos_weight = neg / pos

# Use the best hyperparameters from Optuna
best_params = best_trial.params  # from your Optuna study
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 42,
    "device": "cpu",
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight
})

best_boost_round = 53 # from mlflow

# Train the final model with early stopping on the last historical season
evals_result = {}
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_boost_round,
    verbose_eval=True
)

# PLOT THE BASE TEST CONFUSION MATRIX
plot_confusion_matrix(final_model, dtest, y_test, "Test", threshold=best_avg_threshold)

# Plotting the feature importance
# 'gain' is the most important metric for interpreting feature contribution
plot_feature_importance(final_model, "gain", 50)

# Plot ROC AUC for all three sets
plot_train_val_test_auc(final_model, [dtrain, dval, dtest], [y_train, y_val, y_test])

# Plot Precision-Recall 
plot_precision_recall(final_model, dtest, y_test, "Test")

# plot the calibration curve
plot_calibration_curve(final_model, dtest, y_test, "Test")
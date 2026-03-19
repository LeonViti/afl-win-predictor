"""
NOT FOR USE IN PIPELINE, LOOK TO DELETE LATER
This code an example of a simple training run (no cv)
"""

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
def create_dummy_row(df: pl.DataFrame, cat_cols: list[str]) -> pl.DataFrame:
    """
    Creates a dummy row with weight 0. Changes categorical columns to `UNK` value. 

    Args:
        df: polars dataframe from compute_fixture_features()
        cat_cols: list of categorical columns
    
    Returns: Polars dataframe with additional dummy row.
    """
    # create a dummy row and ensure it is assigned a weight of zero when training
    dummy = (
        df.head(1)
        .with_columns([pl.lit("UNK").alias(c) for c in cat_cols])
        .with_columns(pl.lit(0.0).alias("weight"))  # zero weight
    )

    # Add weight=1 for all real rows
    df = df.with_columns(
        pl.lit(1.0).alias("weight")
    )

    # Append dummy
    df = df.vstack(dummy)

    # convert to categorical for xgboost dmatrix support (in case types change)
    df_final = df.with_columns(
        pl.col(pl.String).cast(pl.Categorical)
    )
    return df_final

def prepare_features(df: pl.DataFrame, cat_cols: list[str]) -> pl.DataFrame:
    """
    Prepares features for modelling. Also builds a dummy row with weight 0.

    Args:
        df: polars dataframe from compute_fixture_features()
        cat_cols: list of categorical columns
    """
    # filter to year afl started (1990)
    df_clean = df.filter(pl.col("year") > 1990)

    df_clean = df_clean.with_columns([ # greate column to indicate a win 
    (pl.col("hscore") > pl.col("ascore")).cast(pl.Int8).alias("win")
    ])

    df_clean = df_clean.select([
        'ateam', 'ateamid', 'hteam', 'hteamid', 'is_final', 'is_grand_final', 'localtime_dt',
        'round', 'winner', 'year', 'venue_location', 'tz_shift_advantage', 
        'hwin_lag1', 'awin_lag1', 'hscore_lag1', 'ascore_lag1', 'hmargin_lag1', 'amargin_lag1',
        "diff_last3_win_rate", "diff_last5_win_rate", "diff_last10_win_rate", "diff_last20_win_rate",
        "diff_last3_avg_score", "diff_last5_avg_score", "diff_last10_avg_score", "diff_last20_avg_score",
        "diff_last3_avg_margin", "diff_last5_avg_margin", "diff_last10_avg_margin", "diff_last20_avg_margin",
        # 'hlast3_win_rate', 'alast3_win_rate', 'hlast5_win_rate', 'alast5_win_rate',
        # 'hlast10_win_rate', 'alast10_win_rate', 'hlast20_win_rate', 'alast20_win_rate',
        'is_interstate_game', "time_of_day", 'is_weekend_game',
        'win'
    ])


    df_clean = df_clean.drop(['ateamid', 'hteamid', 'winner', 'localtime_dt'])

    df_clean = create_dummy_row(df_clean, cat_cols)

    # convert to categorical for xgboost dmatrix support
    df_model = df_clean.with_columns(
        pl.col(pl.String).cast(pl.Categorical)
    )
    return df_model

def handle_unseen_categories(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    cat_cols,
    unk_token: str = "UNK"
) -> pl.DataFrame:
    """
    Replace category values in val_df that do not appear in train_df with unk_token

    Args
        train_df: training df
        val_df: validation df
        cat_cols: list of categorical column names
        unk_token: replacement value for unseen categories

    Returns
        Updated val_df with unseen categories mapped to unk_token.
    """
    # set this internally for convenience

    for col in cat_cols:
        train_cats = set(train_df[col].unique().to_list())
        
        # check for missing column value in validation set, if missing change to dummy value UNK
        val_df = val_df.with_columns(
            pl.when(~pl.col(col).is_in(train_cats))
            .then(pl.lit(unk_token))
            .otherwise(pl.col(col))
            .alias(col)
        )

    return val_df

def create_rolling_forward_walk(
    cv_df:pl.DataFrame,
    window_size:int,
    cat_cols:list[str]
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Create rolling forward walk-forward cross-validation folds.
    
    Args:
        cv_df: DataFrame containing historical seasons (excluding test)
        window_size: range of seasons to train on
        cat_cols: list of categorical columns

    Returns: List of (train_df, val_df) tuples for rolling CV
    """
    # get seasons as a list
    seasons = sorted(cv_df.select("year").unique().to_series().to_list())

    walk_folds = []

    for i in range(window_size, len(seasons)):
        
        train_seasons = seasons[i - window_size : i]
        val_season = seasons[i]

        train_df = cv_df.filter(pl.col("year").is_in(train_seasons))
        
        val_df = cv_df.filter(
            (pl.col("year") == val_season) &
            (pl.col("weight") == 1.0)   # remove dummy row
        )

        # Handle unseen categorical values
        val_df = handle_unseen_categories(train_df, val_df, cat_cols, "UNK")

        # Ensure categorical casting is correct for train and val
        train_df = train_df.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
        val_df = val_df.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))

        walk_folds.append((train_df, val_df))

    return walk_folds

def print_fold_windows(walk_folds:list[tuple[pl.DataFrame, pl.DataFrame]]) -> None:
    """
    Print the train and validation years for each fold.

    Args:
        walk_folds: List of tuples containing (train_df, val_df) for each fold
    """
    for idx, (tr, val) in enumerate(walk_folds):
        print(f"Fold {idx+1}: train years = {tr['year'].unique().to_list()}, val year = {val['year'].unique().to_list()}")


def get_best_accuracy_threshold(
    y: pl.Series,
    y_pred: np.ndarray,
    n_thresholds: int = 200
) -> tuple[np.float64, np.float64]:
    """
    Find the threshold that gives the highest classification accuracy.

    Args:
        y: True labels (Polars Series)
        y_pred: Predicted probabilities (numpy array)
        n_thresholds: Number of thresholds to scan between 0 and 1

    Returns:
        best_threshold: Threshold that maximizes accuracy
        best_accuracy: Accuracy at that threshold
    """
    # convert Polars Series to NumPy array
    y_np = y.to_numpy()

    # select thresholds to check accuracies over
    thresholds = np.linspace(0, 1, n_thresholds)
    
    # Vectorized computation
    y_pred_matrix = (y_pred[None, :] >= thresholds[:, None]).astype(int)
    accuracies = (y_pred_matrix == y_np[None, :]).mean(axis=1)
    
    best_idx = np.argmax(accuracies)
    best_t = thresholds[best_idx]
    best_acc = accuracies[best_idx]

    return best_t, best_acc

def get_best_mlflow_avg_threshold(study, trial_date_code):
    """
    trial_date = '20260308_205622'
    """
    # Best trial
    best_trial = study.best_trial
    print("Best Mean Validation Accuracy:", best_trial.value)
    print("Best Hyperparameters:", best_trial.params)

    # to get the best avg_threshold from MLflow
    best_trial_number = best_trial.number  # Optuna best trial number

    # Get the MLflow run corresponding to that trial
    client = mlflow.tracking.MlflowClient()
    experiment_id = mlflow.get_experiment_by_name("afl_win_predictor").experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = 'trial_{best_trial_number}_{trial_date_code}'"
    )

    best_run = runs[0]  # should be exactly one
    best_avg_threshold = best_run.data.metrics["mean_val_threshold"]

    return best_avg_threshold

#####################
# Code
#####################

path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
df = compute_fixture_features(path)

# prepare features
cat_cols = ["ateam", "hteam", "venue_location", "time_of_day"]
df_model = prepare_features(df, cat_cols)

# This works, but converts to NumPy internally
# TODO: move single run example elsewhere
# train_df, dummy_df = train_test_split(df_model, test_size=0.3, random_state=42)
# val_df, test_df = train_test_split(dummy_df, test_size=0.5, random_state=42)
# Separate final test season
test_df = df_model.filter(pl.col("year") == 2025)
val_df = df_model.filter(pl.col("year") == 2024)
train_df = df_model.filter(pl.col("year") < 2023)

# perform splits for X and y and assign weights
X_train, X_val, X_test = train_df.drop(["win", "weight", "year"]), val_df.drop(["win", "weight", "year"]), test_df.drop(["win", "weight", "year"])
y_train, y_val, y_test = train_df["win"], val_df["win"], test_df["win"]
w_train, w_val, w_test = train_df["weight"], val_df["weight"], test_df["weight"]

# Convert to XGBoost DMatrix form
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test, enable_categorical=True)

y_train_np = y_train.to_numpy()
pos = (y_train_np == 1).sum()
neg = (y_train_np == 0).sum()
scale_pos_weight = neg / pos

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 42,
    "device": "cpu", # or "cuda" if you have a GPU
    "scale_pos_weight": scale_pos_weight 
}

evals_result = {}

# Train with early stopping!
# If the 'dval' AUC doesn't improve for 10 rounds, it stops automatically.
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=20, 
    evals_result=evals_result
)

# PLOT THE BASE TEST CONFUSION MATRIX
plot_confusion_matrix(model, dtest, y_test, "Test")

# PLOT THE TRAINING HISTORY
plot_training_history(model, evals_result)

# Plotting the feature importance
# 'gain' is the most important metric for interpreting feature contribution
plot_feature_importance(model, "gain", 25)

# Plot ROC AUC for all three sets
plot_train_val_test_auc(model, [dtrain, dval, dtest], [y_train, y_val, y_test])

# Plot Precision-Recall 
plot_precision_recall(model, dval, y_val, "Validation")
plot_precision_recall(model, dtest, y_test, "Test")

plot_calibration_curve(model, dval, y_val, "Validation")

###############################
# POST THRESHOLD ADJUSTMENT
###############################

# predict accuracy values for differing thresholds and get the best threshold
y_scores = model.predict(dval) 
thresholds = np.linspace(0, 1, 200)
accuracies = []
for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
accuracies = np.array(accuracies)
best_idx = np.argmax(accuracies)
best_t = thresholds[best_idx]
best_acc = accuracies[best_idx]

get_best_accuracy_threshold(model, dval, y_val)

# plot the accuracy vs threshold to select the threshold to maximise accuracy 
plot_accuracy_vs_threshold(thresholds, accuracies, best_acc, best_t)

# plot the confusion matrix with the best threshold set
plot_confusion_matrix(model, dtest, y_test, "BT Test", best_t)
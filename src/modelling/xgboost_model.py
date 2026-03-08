# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Libraries
import optuna
import mlflow
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

    df_clean = df_clean.select(['ateam', 'ateamid', 'hteam', 'hteamid', 'is_final', 'is_grand_final', 'localtime_dt',
                        'round', 'winner', 'year', 'venue_location', 'tz_shift_advantage',
                        'hlast5_win_rate', 'alast5_win_rate', 'win'])


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

#####################
# Code
#####################

path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
df = compute_fixture_features(path)

# prepare features
cat_cols = ["ateam", "hteam", "venue_location"]
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
plot_feature_importance(model, "gain", 10)

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

# plot the accuracy vs threshold to select the threshold to maximise accuracy 
plot_accuracy_vs_threshold(thresholds, accuracies, best_acc, best_t)

# plot the confusion matrix with the best threshold set
plot_confusion_matrix(model, dtest, y_test, "BT Test", best_t)

############
# mlfow 
############
mlflow.set_experiment("afl_win_predictor")

# --- EXPANDING WALK-FORWARD FOLDS SETUP ---
# prepare features
cat_cols = ["ateam", "hteam", "venue_location"]
df_model = prepare_features(df, cat_cols)

# Separate final test season
test_df = df_model.filter(pl.col("year") == 2025)

# Historical seasons for CV
cv_df = df_model.filter(pl.col("year") < 2025)
seasons = sorted(cv_df["year"].unique())

# Pick 10 folds spaced evenly across the years (adjust as needed)
# fold_indices = np.linspace(3, len(seasons)-1, 20, dtype=int)  # start at 3 to have enough train data
fold_indices = sorted(set(np.linspace(3, len(seasons)-1, 20).astype(int)))
walk_folds = []

for i in fold_indices:
    train_seasons = seasons[:i]
    val_season = seasons[i]
    train_df = cv_df.filter(pl.col("year").is_in(train_seasons))
    val_df = cv_df.filter(pl.col("year") == val_season)
    # handle unseen values 
    val_df = handle_unseen_categories(train_df, val_df, cat_cols, "UNK")

    # if dummy_row is present in the validation set, remove it
    val_df = val_df.filter(pl.col("weight") == 1.0)

    walk_folds.append((train_df, val_df))

# Check fold years
# for idx, (tr, val) in enumerate(walk_folds):
#     print(f"Fold {idx+1}: train years = {tr['year'].unique().to_list()}, val year = {val['year'].unique().to_list()}")

# --- ROLLING WALK FORWARD SET-UP ---
mlflow.set_experiment("afl_win_predictor")

cat_cols = ["ateam", "hteam", "venue_location"]
df_model = prepare_features(df, cat_cols)

# Separate final test season
test_df = df_model.filter(pl.col("year") == 2025)

# get seasons aside from the last season for training
cv_df = df_model.filter(pl.col("year") < 2025)

# get seasons as a list
seasons = sorted(cv_df.select("year").unique().to_series().to_list())

# create rolling window walk folds
walk_folds = create_rolling_forward_walk(cv_df, 5, cat_cols)

# print fold windows
for idx, (tr, val) in enumerate(walk_folds):
    print(f"Fold {idx+1}: train years = {tr['year'].unique().to_list()}, val year = {val['year'].unique().to_list()}")

# TODO: add threshold selection to the pipeline
def objective(trial):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
        "device": "cpu",
        "tree_method": "hist",  # recommended
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    fold_aucs = []
    best_rounds = []
    
    for train_df, val_df in walk_folds:

        X_train = train_df.drop(["win", "year", "weight"])
        y_train = train_df["win"]
        w_train = train_df["weight"]

        X_val = val_df.drop(["win", "year", "weight"])
        y_val = val_df["win"]
        w_val = val_df["weight"]

        # Class imbalance (exclude dummy rows)
        y_effective = train_df.filter(pl.col("weight") == 1.0)["win"].to_numpy()
        pos = (y_effective == 1).sum()
        neg = (y_effective == 0).sum()
        params["scale_pos_weight"] = neg / pos

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, enable_categorical=True)

        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False,
        )

        y_val_pred = model.predict(dval)
        fold_aucs.append(roc_auc_score(y_val, y_val_pred))
        best_rounds.append(model.best_iteration + 1)

    mean_auc = float(np.mean(fold_aucs))
    mean_best_rounds = int(np.mean(best_rounds))

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("mean_val_auc", mean_auc)
        mlflow.log_metric("num_folds", len(walk_folds))
        mlflow.log_metric("mean_best_num_boost_rounds", mean_best_rounds)

        # log last fold model only as reference
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="afl_xgb_model_last_fold"
        )

    return mean_auc

# TODO: create optuna plots in mlflow

# Create the study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, n_jobs=-1)  # n_trials can be larger
# Best trial
best_trial = study.best_trial
print("Best validation AUC:", best_trial.value)
print("Best hyperparameters:", best_trial.params)

# TODO: evaluate hyperparameter tuning with Optuna

# LOAD BEST MODEL AND EVALUATE PERFORMANCE TODO: consider turning this section to evaluate.py
model_uri = "runs:/66ae62b956fc49778e7602575598039b/afl_xgb_model_last_fold_5"

loaded_model = mlflow.xgboost.load_model(model_uri)

# NOTE: ensure the dummy row is removed when evaluating

# PLOT THE BASE TEST CONFUSION MATRIX
plot_confusion_matrix(loaded_model, dtest, y_test, "Test")

# Plotting the feature importance
# 'gain' is the most important metric for interpreting feature contribution
plot_feature_importance(loaded_model, "gain", 10)

# Plot ROC AUC for all three sets
plot_train_val_test_auc(loaded_model, [dtrain, dval, dtest], [y_train, y_val, y_test])

# Plot Precision-Recall 
plot_precision_recall(loaded_model, dval, y_val, "Validation")
plot_precision_recall(loaded_model, dtest, y_test, "Test")

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

# plot the accuracy vs threshold to select the threshold to maximise accuracy 
plot_accuracy_vs_threshold(thresholds, accuracies, best_acc, best_t)

# plot the confusion matrix with the best threshold set
plot_confusion_matrix(model, dtest, y_test, "BT Test", best_t)

# delete mlflow runs
# Get the experiment ID
exp = mlflow.get_experiment_by_name("afl_win_predictor")
experiment_id = exp.experiment_id

# Delete all runs
for run_info in mlflow.list_run_infos(experiment_id):
    mlflow.delete_run(run_info.run_id)

print("All runs deleted for experiment:", experiment_id)
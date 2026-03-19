"""
This file runs the hyperparameter search using optuna. 
MLFlow is used for parameter logging. 
go `mlflow ui` to check the runs. 
"""

# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"/home/lv/Documents/projects/afl-win-predictor")

#####################
# Libraries
#####################
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


################
# Code
################
def main():
    path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
    df = compute_fixture_features(path)

    # prepare features
    cat_cols = ["ateam", "hteam", "venue_location", "time_of_day"]
    df_model = prepare_features(df, cat_cols)

    # --- ROLLING WALK FORWARD SET-UP ---
    mlflow.set_experiment("afl_win_predictor")

    cat_cols = ["ateam", "hteam", "venue_location", "time_of_day"]
    df_model = prepare_features(df, cat_cols)

    # Separate final test season
    test_df = df_model.filter(pl.col("year") == 2025)

    # get seasons aside from the last season for training
    cv_df = df_model.filter(pl.col("year") < 2025)

    # create rolling window walk folds
    walk_folds = create_rolling_forward_walk(cv_df, 5, cat_cols)

    # print fold windows
    print_fold_windows(walk_folds)

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
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),        
        }

        fold_aucs = []
        fold_thresholds = []
        fold_accuracies = []
        fold_best_rounds = []

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

            # make prediction on the validation set
            y_val_pred = model.predict(dval)

            # append metrics
            best_t, best_acc = get_best_accuracy_threshold(y_val, y_val_pred)
            fold_aucs.append(roc_auc_score(y_val, y_val_pred))
            fold_thresholds.append(best_t)
            fold_accuracies.append(best_acc)
            fold_best_rounds.append(model.best_iteration + 1)  # add 1 because best_iteration is zero-indexed

        # calculate mean metrics across folds
        mean_auc = float(np.mean(fold_aucs))
        mean_threshold = float(np.mean(fold_thresholds))
        mean_accuracy = float(np.mean(fold_accuracies))
        mean_best_rounds = int(np.mean(fold_best_rounds))

        # define unique run name
        run_name = f"trial_{trial.number}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(nested=True, run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metric("mean_val_auc", mean_auc)
            mlflow.log_metric("mean_val_threshold", mean_threshold)
            mlflow.log_metric("mean_val_accuracy", mean_accuracy)
            mlflow.log_metric("mean_best_num_boost_rounds", mean_best_rounds)
            mlflow.log_metric("num_folds", len(walk_folds))

            for i, (auc, acc, t, rounds) in enumerate(zip(fold_aucs, fold_accuracies, fold_thresholds, fold_best_rounds)):
                mlflow.log_metric(f"fold_{i+1}_auc", auc)
                mlflow.log_metric(f"fold_{i+1}_accuracy", acc)
                mlflow.log_metric(f"fold_{i+1}_threshold", t)
                mlflow.log_metric(f"fold_{i+1}_best_rounds", rounds)

        return mean_accuracy

    # TODO: create optuna plots in mlflow

    # Create the study and optimize
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=300, n_jobs=-1)  # n_trials can be larger
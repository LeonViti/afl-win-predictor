# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Libraries
import mlflow
import mlflow.xgboost
import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score

from src.config import PROJECT_ROOT
from src.features.fixture_features import compute_fixture_features

#####################
# Functions
#####################
def plot_confusion_matrix(
    model:xgb.Booster,
    dmatrix:xgb.DMatrix,
    y:pl.Series, 
    set_name: Optional[str] = "",
    threshold: Optional[float] = 0.5
) -> None:
    """
    Plots a confusion matrix. 

    Args:
        model: Trained binary classification model with a `.predict()` method. 
        dmatrix: Dmatrix dataframe. E.g. dtest, dval, dtrain. 
        y: True labels for the set of interest. E.g. y_test, y_val, y_train. 
        threshold: threshold to filter classification cut-off (default is 0.5)
    """

    # get proba predictions (0.0 to 1.0)
    y_probs = model.predict(dmatrix)

    # convert to binary classes (1 if prob > threshold [default 0.5] else 0)
    y_preds = (y_probs > threshold).astype(int)

    # compute the confusion matrix
    cm = confusion_matrix(y, y_preds)

    # plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Loss", "Win"])
    disp.plot(cmap="Purples", values_format="d")
    plt.title(f"{set_name} Confusion Matrix (Threshold={threshold:.2f})")
    plt.show()

def plot_training_history(evals_result: dict) -> None:
    """
    Plots the training history to compare train and validation performance during a run. 

    Args:
        evals_result: dictionary containing the train and val results per iteration. 
    """
    train_auc = evals_result['train']['auc']
    val_auc = evals_result['validation']['auc']
    epochs = range(len(train_auc))
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_auc, label='Train AUC', color='blue')
    plt.plot(epochs, val_auc, label='Validation AUC', color='orange')
    # Add a vertical line where early stopping happened
    plt.axvline(x=model.best_iteration, color='red', linestyle='--', label='Best Iteration')
    plt.title('AFL Model Training History')
    plt.xlabel('Number of Iterations')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(
    model: xgb.Booster,
    importance_type: Optional[str] = 'gain',
    max_num_features: Optional[int] = 10
) -> None:
    """
    Plots the top features by importance from an XGBoost model.

    Args:
        model: Trained XGBoost model.
        importance_type: Type of feature importance to plot. Default is 'gain'
        max_num_features: Maximum number of top features to display. Default is 10.

    Displays:
        A matplotlib bar plot showing feature importance.
    """

    # Get feature importance as a dictionary
    importance_dict = model.get_score(importance_type=importance_type)

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:max_num_features] # filter for max_feats

    features, scores = zip(*sorted_features) # extract data from dict as tuples

    # Plot
    plt.figure(figsize=(10,6))
    bars = plt.barh(
        features[::-1],  # reverse so top features appear on top
        scores[::-1],
        color='darkorchid',
        edgecolor='black',
        linewidth=0.5,
        height=0.5,
        zorder=2
    )
    # Add value labels at the end of each bar
    for bar, score in zip(bars, scores[::-1]):
        plt.text(
            bar.get_width() + max(scores)*0.01,  # slightly past the bar end
            bar.get_y() + bar.get_height()/2,
            f"{score:.3f}",  # rounded to 3 dp
            va='center',
            ha='left',
            fontsize=9
        )

    plt.xlabel(f"Importance Score ({importance_type})")
    plt.title(f"Feature Importance ({importance_type})")
    plt.grid(alpha=1, axis='x', zorder=1)
    plt.show()

def plot_train_val_test_auc(dmats:list[xgb.DMatrix], ys:list[pl.Series]) -> None:
    """
    Plot the ROC AUC for the models performance on the train, validation, and test sets. 

    Args:
        dmats: list of dmatrices for train, val, and test ([dtrain, dval, dtest])
        ys: True labels for the train, val, and test sets. E.g. y_test, y_val, y_train. 
    """
    plt.figure(figsize=(8,8))

    colors = ["mediumpurple", "darkorchid", "rebeccapurple"]  # shades of purple
    labels = ["Train", "Validation", "Test"]

    for y_true, y_scores, label, color in zip(ys, [model.predict(d) for d in dmats], labels, colors):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0,1], [0,1], color='lightgray', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Train, Validation, Test ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_precision_recall(
    model: xgb.Booster, 
    dmatrix: xgb.DMatrix,
    y:pl.Series, 
    set_name: Optional[str] = ""
) -> None:
    """
    Plots the precision recall curve of a model. 

    Args:
        model: model with a .predict method.
        dmatrix: Dmatrix dataframe. E.g. dtest, dval, dtrain. 
        y: True labels for the set of interest. E.g. y_test, y_val, y_train. 
    """
    # get predicted probabilities
    y_scores = model.predict(dmatrix) 
    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    ap_score = average_precision_score(y, y_scores)

    # Plot 
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color="darkorchid")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{set_name} Precision-Recall Curve (Avg. Precision = {ap_score:.3f})")
    plt.grid(alpha=0.3)
    plt.show()

def plot_accuracy_vs_threshold(thresholds: np.ndarray, accuracies: np.ndarray, best_acc: np.float64, best_t: np.float64) -> None:
    """
    Creates accuracy vs threshold plot from the Validation set.

    Args:
        thresholds (np.ndarray): Array of threshold values used for evaluation.
        accuracies (np.ndarray): Accuracy values corresponding to each threshold.
        best_acc (np.float64): Maximum accuracy achieved.
        best_t (np.float64): Threshold that gives the maximum accuracy.
    """
    # plot the graph
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(thresholds, accuracies, label=f"BA = {best_acc:.1%}", color='darkorchid', zorder=3)
    plt.axvline(best_t, linestyle=":", color='k', alpha=0.4, label=f"BT = {best_t:.3f}", zorder=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    print("Best threshold:", best_t)
    print("Best accuracy:", best_acc)

#####################
# Code
#####################

path = PROJECT_ROOT / 'data/complete_datasets/squiggle_fixture_all_seasons.parquet'
df = compute_fixture_features(path)

df_clean = df.with_columns([ # greate column to indicate a win 
   (pl.col("hscore") > pl.col("ascore")).cast(pl.Int8).alias("win")
])

df_clean = df_clean.select(['ateam', 'ateamid', 'hteam', 'hteamid', 'is_final', 'is_grand_final', 'localtime_dt',
                      'round', 'winner', 'year', 'venue_location', 'tz_shift_advantage',
                      'hlast5_win_rate', 'alast5_win_rate', 'win'])


df_clean = df_clean.drop(['ateamid', 'hteamid', 'winner', 'localtime_dt'])

# convert to categorical for xgboost dmatrix support
df_clean = df_clean.with_columns(
    pl.col(pl.String).cast(pl.Categorical)
)

# This works, but converts to NumPy internally
# train_df, dummy_df = train_test_split(df_clean, test_size=0.3, random_state=42)
# val_df, test_df = train_test_split(dummy_df, test_size=0.5, random_state=42)
# Separate final test season
test_df = df_clean.filter(pl.col("year") == 2025)
val_df = df_clean.filter(pl.col("year") == 2024)
train_df = df_clean.filter(pl.col("year") < 2024)

# perform splits for X and y
X_train, X_val, X_test = train_df.drop("win"), val_df.drop("win"), test_df.drop("win")
y_train, y_val, y_test = train_df["win"], val_df["win"], test_df["win"]

# Convert to XGBoost DMatrix form
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

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

# NOTE: below uncommented code is for hyperparam tuning. 
# Run CV (for hyperparameter tuning)
# cv_results = xgb.cv(
#     params=params,
#     dtrain=dtrain,
#     num_boost_round=2000,
#     nfold=5,
#     metrics="auc",
#     early_stopping_rounds=100,
#     seed=42,
#     verbose_eval=50,  # prints progress
#     as_pandas=True
# )

# Train final model using best number of rounds
# model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=best_nrounds,
#     evals=[(dtrain, "train"), (dval, "validation")],
#     early_stopping_rounds=20,
#     evals_result=evals_result
# )

# PLOT THE BASE TEST CONFUSION MATRIX
plot_confusion_matrix(model, dtest, y_test, "Test")

# PLOT THE TRAINING HISTORY
plot_training_history(evals_result)

# Plotting the feature importance
# 'gain' is the most important metric for interpreting feature contribution
plot_feature_importance(model, "gain", 10)

# Plot ROC AUC for all three sets
plot_train_val_test_auc([dtrain, dval, dtest], [y_train, y_val, y_test])

# Plot Precision-Recall 
plot_precision_recall(model, dval, y_val, "Validation")
plot_precision_recall(model, dtest, y_test, "Test")

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
# mlfow 3
############
import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("afl_win_predictor")

# --- WALK-FORWARD FOLDS SETUP ---
# Separate final test season
test_df = df_clean.filter(pl.col("year") == 2025)

# Historical seasons for CV
cv_df = df_clean.filter(pl.col("year") < 2025)
seasons = sorted(cv_df["year"].unique())

# Pick 10 folds spaced evenly across the years (adjust as needed)
fold_indices = np.linspace(3, len(seasons)-1, 10, dtype=int)  # start at 3 to have enough train data
walk_folds = []

for i in fold_indices:
    train_seasons = seasons[:i]
    val_season = seasons[i]
    train_df = cv_df.filter(pl.col("year").is_in(train_seasons))
    val_df   = cv_df.filter(pl.col("year") == val_season)
    walk_folds.append((train_df, val_df))

# # Check fold years
# for idx, (tr, val) in enumerate(walk_folds):
#     print(f"Fold {idx+1}: train years = {tr['year'].unique().to_list()}, val year = {val['year'].unique().to_list()}")

def objective(trial):
    # Suggest hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
        "device": "cpu",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0)
    }

    fold_aucs = []

    # Walk-forward CV
    for fold_idx, (train_df, val_df) in enumerate(walk_folds):
        X_train, y_train = train_df.drop("win"), train_df["win"]
        X_val, y_val     = val_df.drop("win"), val_df["win"]

        # Handle class imbalance
        y_train_np = y_train.to_numpy()
        pos = (y_train_np == 1).sum()
        neg = (y_train_np == 0).sum()
        params["scale_pos_weight"] = neg / pos

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval   = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=20,
            verbose_eval=False,
            evals_result=evals_result
        )

        # Fold AUC for diagnostics
        y_val_pred = model.predict(dval)
        fold_auc = roc_auc_score(y_val, y_val_pred)
        fold_aucs.append(fold_auc)

    # Mean AUC across folds is the single Optuna objective
    mean_auc = np.mean(fold_aucs)

    # Log trial in MLflow (only mean AUC)
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("mean_val_auc", mean_auc)
        mlflow.log_metric("num_folds", len(walk_folds))
        mlflow.log_metric("best_num_boost_rounds", model.best_iteration + 1)

        # Log final model of last fold for reference
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="afl_xgb_model_last_fold",
        )

    return mean_auc  # Optuna maximizes this

# Create the study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, n_jobs=1)  # n_trials can be larger

# Best trial
best_trial = study.best_trial
print("Best validation AUC:", best_trial.value)
print("Best hyperparameters:", best_trial.params)

# Path to the model in the run
model_uri = "runs:/0c5b080bac5a4b58a06b1fcb6c26a1d6/afl_xgb_model"

loaded_model = mlflow.xgboost.load_model(model_uri)

# Use it for predictions
y_probs = loaded_model.predict(dtest)

# plot thresholds vs accuracies
plot_accuracy_vs_threshold(loaded_model, dval, y_val)

# CM WITH THRESHOLD
plot_cm_with_threshold(loaded_model, dval, y_val, dtest, y_test)
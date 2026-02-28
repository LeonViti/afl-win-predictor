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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score

from src.config import PROJECT_ROOT
from src.features.fixture_features import compute_fixture_features

#####################
# Functions
#####################
def plot_accuracy_vs_threshold(model: xgb.Booster, dval: xgb.DMatrix, y_val:pl.Series) -> None:
    """
    Creates accuracy vs threshold plot from the Validation set.

    Args:
        model: Trained binary classification model with a `.predict()` method. 
        dval: Validation dataframe. 
        y_val: True labels for the validation set.
    """
    # predict accuracy values for differing thresholds
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

def plot_cm_with_threshold(model: xgb.Booster, dval: xgb.DMatrix, y_val:pl.Series, dtest:xgb.DMatrix, y_test:pl.Series):
    # calc the best threshold value fr val set TODO: consider moving this to the code block
    """
    Plots the confusion matrix with the threshold that maximises accuracy.

    Args:
        model: Trained binary classification model with a `.predict()` method. 
        dval: DMatrix Validation dataframe. 
        y_val: True labels for the validation set.
        dtest: DMatrix test dataframe. 
        y_test: True labels for the test set.
    """
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

    # get test set predictions and plot the confusion matrix
    y_probs = model.predict(dtest)
    y_preds = (y_probs > best_t).astype(int)
    cm = confusion_matrix(y_test, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Loss", "Win"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("AFL Win Predictor: Confusion Matrix")
    plt.show()

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


df_clean = df_clean.drop(['ateamid', 'hteamid', 'winner', 'year', 'localtime_dt'])

# convert to categorical for xgboost dmatrix support
df_clean = df_clean.with_columns(
    pl.col(pl.String).cast(pl.Categorical)
)

# This works, but converts to NumPy internally
train_df, dummy_df = train_test_split(df_clean, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(dummy_df, test_size=0.5, random_state=42)

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

# PLOT THE CONFUSION MATRIX
# 1. Get probability predictions (0.0 to 1.0)
y_probs = model.predict(dtest)

# 2. Convert to binary classes (1 if prob > 0.5 else 0)
y_preds = (y_probs > 0.5).astype(int)

# 3. Compute the confusion matrix
cm = confusion_matrix(y_test, y_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Loss", "Win"])
disp.plot(cmap="Blues", values_format="d")

plt.title("AFL Win Predictor: Confusion Matrix")
plt.show()

# Extract metrics
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


# Plotting the feature importance
# 'gain' is the most important metric for interpreting feature contribution
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='gain', max_num_features=10)
plt.title("AFL Feature Importance (Gain)")
plt.show()

# Plotting the ROC AUC
y_probs = model.predict(dtest)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('AFL Win Predictor: ROC Curve (Test Set)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Precision-Recall values
# Get predicted probabilities
y_scores = model.predict(dval)  # already probabilities!
precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
ap_score = average_precision_score(y_val, y_scores)
# Plot (single plot, no custom colors)
plt.figure()
plt.plot(recall, precision)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (Avg. Precision = {ap_score:.3f})")
plt.grid(alpha=0.3)
plt.show()

# plot thresholds vs accuracies
plot_accuracy_vs_threshold(model, dval, y_val)

# CM WITH THRESHOLD
plot_cm_with_threshold(model, dval, y_val, dtest, y_test)

# mlflow
mlflow.set_experiment("afl_win_predictor")

with mlflow.start_run(run_name="xgb_train"):
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_param("num_boost_round", model.best_iteration + 1)
    
    # Log metrics
    mlflow.log_metric("train_auc", train_auc[-1])
    mlflow.log_metric("val_auc", val_auc[-1])
    
    # Log model
    mlflow.xgboost.log_model(
        xgb_model=model, 
        name="afl_xgb_model",
        registered_model_name="AFLWinPredictor"
    )
    
    # Optionally log plots as artifacts
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_auc, label="Train AUC")
    plt.plot(epochs, val_auc, label="Validation AUC")
    plt.axvline(x=model.best_iteration, color='red', linestyle='--', label='Best Iteration')
    plt.title("Training History")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    mlflow.log_artifact("training_history.png")
    plt.close()

# TODO Perform Target Encoding for high cardinality features
import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("afl_win_predictor")

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42,
        "device": "cpu",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_uniform("gamma", 0.0, 5.0)
    }

    # Train model with early stopping
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

    # Use the best validation AUC as the objective
    val_auc = evals_result['validation']['auc'][model.best_iteration]
    
    # Log everything in MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("best_iteration", model.best_iteration + 1)
        
        # Save training history plot
        train_auc = evals_result['train']['auc']
        val_auc_history = evals_result['validation']['auc']
        epochs = range(len(train_auc))
        plt.figure(figsize=(8,6))
        plt.plot(epochs, train_auc, label='Train AUC')
        plt.plot(epochs, val_auc_history, label='Validation AUC')
        plt.axvline(model.best_iteration, color='red', linestyle='--', label='Best Iteration')
        plt.title("Training History")
        plt.xlabel("Iteration")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_history.png")
        mlflow.log_artifact("training_history.png")
        plt.close()
        
        # Log XGBoost model
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="afl_xgb_model",
            registered_model_name="AFLWinPredictor"
        )
    
    return val_auc  # Optuna maximizes this

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
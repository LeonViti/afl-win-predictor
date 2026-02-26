# FOR INTERACTIVE SESSION comment out later
import os
os.chdir(r"C:\Users\leon_\Documents\personal_projects\afl-win-predictor")

# Libraries
import mlflow
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from src.config import PROJECT_ROOT
from src.features.fixture_features import compute_fixture_features

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

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 42,
    "device": "cpu" # or "cuda" if you have a GPU
}

evals_result = {}

# Train with early stopping!
# If the 'dval' AUC doesn't improve for 10 rounds, it stops automatically.
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=10, 
    evals_result=evals_result
)

print(evals_result['train']['auc'][:5])

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


# Plotting the importance
# 'gain' is the most important metric for interpreting feature contribution
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='gain', max_num_features=10)
plt.title("AFL Feature Importance (Gain)")
plt.show()

# 1. Get probabilities for the test set
y_probs = model.predict(dtest)

# 2. Calculate False Positive Rate, True Positive Rate, and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 3. Plotting
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


# CODE
mlflow.set_experiment("afl_win_predictor")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Log or run a model using uv as the manager
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MyModel(),
    env_manager="uv"  # <--- Faster environment setup
)

# TODO Perform Target Encoding for high cardinality features
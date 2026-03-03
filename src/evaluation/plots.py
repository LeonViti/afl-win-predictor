"""
This file contains all functions for plotting graphs and figures. 
"""
# Libraries
import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    precision_recall_curve,
    average_precision_score,
) 

# Functions
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

def plot_training_history(model:xgb.Booster ,evals_result: dict) -> None:
    """
    Plots the training history to compare train and validation performance during a run. 

    Args:
        model: Trained binary classification model with a `.best_iteration()` method. 
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

def plot_train_val_test_auc(model, dmats:list[xgb.DMatrix], ys:list[pl.Series]) -> None:
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

def plot_calibration_curve(
    model:xgb.Booster, 
    dmatrix:xgb.DMatrix, 
    y:pl.Series,
    set_name: Optional[str] = "",
) -> None:
    """
    Plot the calibration curve for the model.

    Args:
        model: Trained XGBoost model.
        dmatrix: Dmatrix dataframe. E.g. dtest, dval, dtrain. 
        y: True labels for the set of interest. E.g. y_test, y_val, y_train. 
    """
    y_scores = model.predict(dmatrix)
    prob_true, prob_pred = calibration_curve(y, y_scores, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1],[0,1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title(f'{set_name} Calibration curve')
    plt.legend()
    plt.grid(True)
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
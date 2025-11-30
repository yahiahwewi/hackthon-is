"""
Module: Model Evaluation
========================
Purpose:
    This module handles the performance assessment of the trained model and calculates 
    critical decision thresholds.

Objective:
    - Calculate AUC-ROC (Area Under the Curve) to measure the model's ability to distinguish classes.
    - Determine the 'Optimal Threshold' that maximizes the F1-Score (balance of Precision and Recall).
    - Generate a Classification Report based on this optimal threshold, not the default 0.5.

Goal:
    To ensure the model is evaluated fairly on imbalanced data and to provide a dynamic 
    decision boundary that can be adjusted for 'Smart Prediction' in the final application.
"""
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and finds the optimal threshold.
    Returns: AUC, Optimal Threshold, Classification Report
    """
    # Get probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # AUC
    auc = roc_auc_score(y_test, y_probs)
    
    # Optimal Threshold (Max F1)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions at optimal threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
    report = classification_report(y_test, y_pred_optimal)
    
    return auc, optimal_threshold, report

"""
Module: Model Training
======================
Purpose:
    This module constructs the final training pipeline, integrating preprocessing with 
    imbalance handling and the classifier. We use Random Forest and SMOTE.

Objective:
    - Address the severe Class Imbalance (Stroke cases are rare) using SMOTE (Synthetic Minority Over-sampling Technique).
    - Train a robust Random Forest Classifier that is less prone to overfitting than single decision trees.
    - Encapsulate the entire flow (Preprocessing -> SMOTE -> Model) into a single Pipeline object.

Goal:
    To build a high-performance predictive model that doesn't just memorize the majority class (Healthy) 
    but actively learns to identify the minority class (Stroke) with high sensitivity.
"""
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def build_pipeline(preprocessor, random_state=42):
    """
    Builds the full training pipeline with SMOTE and Classifier.
    """
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))
    ])
    
    return model

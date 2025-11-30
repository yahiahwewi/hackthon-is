"""
Module: Feature Preprocessing
=============================
Purpose:
    This module defines the transformation pipelines for numerical and categorical data.
    We use Scikit-Learn's ColumnTransformer and Imblearn's Pipeline.

Objective:
    - Handle missing values in 'bmi' using KNNImputer (finding similar patients) instead of simple mean/median.
    - Scale numerical features (Age, Glucose, BMI) to normalize distributions.
    - One-Hot Encode categorical features (Gender, Work Type, etc.) for model compatibility.

Goal:
    To transform raw clinical data into a mathematical format that the Machine Learning model 
    can process effectively, while preserving the statistical relationships via smart imputation.
"""
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline

def get_preprocessor(numerical_cols, categorical_cols):
    """
    Returns a ColumnTransformer for preprocessing.
    """
    # Numerical: Impute missing BMI with KNN, then Scale
    numerical_transformer = ImbPipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # Categorical: One-Hot Encode
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

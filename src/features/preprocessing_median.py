"""
Module: Preprocessing – Median Imputer (simple baseline)
=====================================================
Purpose:
    Provide a preprocessing pipeline that uses a **median** imputer for the
    numeric features. This is the fastest alternative and works well when the
    missing values are assumed to be randomly distributed.

Why Median?
    • Very low computational cost – O(N) instead of O(N²) for KNN.
    • Robust to outliers (median is less affected than mean).
    • Serves as a strong baseline to compare against more sophisticated
      approaches such as KNN or Iterative imputation.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Keep the same column definitions as the KNN version for consistency
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
    "hypertension",
    "heart_disease",
]

def get_preprocessor():
    """Return a ColumnTransformer that applies Median imputation + scaling.
    The object can be plugged directly into the training pipeline.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

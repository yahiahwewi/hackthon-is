"""
Module: Preprocessing – KNN Imputer (baseline)
================================================
Purpose:
    Provide the original preprocessing pipeline that uses a K‑Nearest‑Neighbors
    imputer for the numeric features (age, avg_glucose_level, bmi).

Why KNN?
    • Preserves local relationships – BMI is estimated from the 5 most similar
      patients based on the other numeric variables.
    • Works well when missingness is not completely random (the dataset shows
      a clear correlation between BMI, age and glucose).
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------
# Column groups – keep them identical across all preprocessing variants
# ---------------------------------------------------------------------
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
    """Return a ColumnTransformer that applies KNN imputation + scaling.

    The returned object can be used directly in an imbalanced‑learn pipeline
    (see `src/models/train_model.py`).
    """
    numeric_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
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

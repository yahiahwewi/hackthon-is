"""
Factory for selecting a preprocessing pipeline.

Usage
-----
>>> from src.features.preprocessing_factory import get_preprocessor
>>> preproc = get_preprocessor("knn")      # KNN version (baseline)
>>> preproc = get_preprocessor("median")   # Median version
"""

from importlib import import_module

def get_preprocessor(method: str = "knn"):
    """
    Parameters
    ----------
    method : str
        One of {"knn", "median"} (case‑insensitive).

    Returns
    -------
    ColumnTransformer
        The preprocessing pipeline defined in the corresponding module.
    """
    method = method.lower()
    if method == "knn":
        mod = import_module("src.features.preprocessing_knn")
    elif method == "median":
        mod = import_module("src.features.preprocessing_median")
    else:
        raise ValueError(f"Unsupported preprocessing method: {method}")

    return mod.get_preprocessor()

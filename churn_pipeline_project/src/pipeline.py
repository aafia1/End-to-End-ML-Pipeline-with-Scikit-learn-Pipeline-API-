
"""
pipeline.py — defines the preprocessing pipeline, candidate estimators, and hyperparameter grids.
"""
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# All non-numeric, non-target, non-id columns are treated as categoricals by default.
EXCLUDE_COLS = {"customerID", "Churn"}

def split_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric = [c for c in cols if c in NUMERIC_COLS]
    categoricals = [c for c in cols if c not in numeric]
    return numeric, categoricals

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def build_estimators_and_grid(random_state: int = 42):
    logreg = LogisticRegression(max_iter=500, n_jobs=None, random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)
    param_grid = [
        {
            "clf": [logreg],
            "clf__C": [0.1, 1.0, 3.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        {
            "clf": [rf],
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
        },
    ]
    return param_grid

def build_full_pipeline(preprocessor) -> Pipeline:
    # The final classifier placeholder is specified in GridSearchCV via param_grid
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    return pipe

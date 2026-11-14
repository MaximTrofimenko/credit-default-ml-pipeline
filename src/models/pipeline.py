"""Общий конвейер предобработки и модели."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def make_pipeline(columns, model):
    numeric = [name for name in columns if name not in CATEGORICAL]
    numeric_steps = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    category_steps = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", category_steps, CATEGORICAL)]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])

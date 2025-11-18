import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator

from .features import FeatureEngineering


def build_preprocessing_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_exclude=np.number)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_selector),
            ("cat", categorical_transformer, categorical_selector),
        ],
    )
    return preprocessor


def build_full_pipeline(model: BaseEstimator, verbose: bool = True) -> Pipeline:
    feature_engineering = FeatureEngineering(verbose=verbose)
    preprocessing = build_preprocessing_pipeline()

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    return full_pipeline

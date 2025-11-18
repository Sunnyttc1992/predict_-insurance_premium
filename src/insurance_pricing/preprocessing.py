from unicodedata import numeric
from column_transformer import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUM_FEATURES ,CAT_FEATURES
from .features import FeatureEngineering

def build_preprocessing_pipeline(verbose: bool = True) -> Pipeline:
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_FEATURES),
            ('cat', categorical_transformer, CAT_FEATURES)
        ]
    )
    return preprocessor

def build_full_pipeline(verbose: bool = True) -> Pipeline:

    feature_engineering = FeatureEngineering(verbose=verbose)
    preprocessing = build_preprocessing_pipeline(verbose=verbose)

    full_pipeline = Pipeline(steps=[
        ('feature_engineering', feature_engineering),
        ('preprocessing', preprocessing),
        ("model",model)
    ])

    return full_pipeline

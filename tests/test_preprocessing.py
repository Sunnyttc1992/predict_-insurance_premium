import pandas as pd
from sklearn.linear_model import LinearRegression

from insurance_pricing.preprocessing import (
    build_full_pipeline,
    build_preprocessing_pipeline,
)


def _sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40],
            "bmi": [22.5, 30.1],
            "children": [0, 2],
            "sex": ["male", "female"],
            "smoker": ["no", "yes"],
            "region": ["northwest", "southeast"],
        }
    )


def test_preprocessing_pipeline_includes_feature_engineering():
    pipeline = build_preprocessing_pipeline(verbose=False)
    step_names = [name for name, _ in pipeline.steps]

    assert step_names[0] == "feature_engineering"
    assert step_names[-1] == "preprocessing"

    pipeline.fit(_sample_features(), [0, 1])
    transformed = pipeline.transform(_sample_features())

    assert transformed.shape[0] == 2


def test_full_pipeline_keeps_feature_engineering_step():
    pipeline = build_full_pipeline(LinearRegression(), verbose=False)
    step_names = [name for name, _ in pipeline.steps]

    assert step_names[0] == "feature_engineering"
    assert step_names[-1] == "model"

    pipeline.fit(_sample_features(), [1000, 2000])
    predictions = pipeline.predict(_sample_features())

    assert predictions.shape == (2,)

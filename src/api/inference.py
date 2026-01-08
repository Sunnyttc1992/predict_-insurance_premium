import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml


# Utility: discover model name from config (fallback to literal name)
def _discover_paths():
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "configs" / "model_config.yaml"
    model_name = "insurance_price_model"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as fh:
                cfg = yaml.safe_load(fh)
            model_name = cfg.get("model", {}).get("name", model_name)
        except Exception:
            # fall back to default name
            pass

    model_path = repo_root / "models" / "trained" / f"{model_name}.pkl"
    preprocessor_path = repo_root / "models" / "trained" / "preprocessor.pkl"
    return str(model_path), str(preprocessor_path)


MODEL_PATH, PREPROCESSOR_PATH = _discover_paths()


def _load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {MODEL_PATH}: {e}")
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading preprocessor from {PREPROCESSOR_PATH}: {e}")
    return model, preprocessor


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same derived features as `src/features/feature_engineer.py`.

    Expected base columns (from insurance dataset): `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
    """
    df = df.copy()

    # Age group buckets
    try:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
    except KeyError:
        raise KeyError("Input data must contain 'age' column")

    # BMI categories
    try:
        df["bmi_category"] = pd.cut(
            df["bmi"], bins=[0, 18.5, 25, 30, 100],
            labels=["underweight", "normal", "overweight", "obese"],
        )
    except KeyError:
        raise KeyError("Input data must contain 'bmi' column")

    # Smoker flag
    if "smoker" not in df.columns:
        raise KeyError("Input data must contain 'smoker' column")
    df["is_smoker"] = (df["smoker"] == "yes").astype(int)

    # High BMI indicator
    df["high_bmi"] = (df.get("bmi", 0) >= 30).astype(int)

    # Interactions
    df["age_smoker_interaction"] = df.get("age", 0) * df["is_smoker"]
    df["bmi_smoker_interaction"] = df.get("bmi", 0) * df["is_smoker"]

    # Family size indicators
    df["family_size"] = df.get("children", 0) + 1
    df["large_family"] = (df["family_size"] >= 4).astype(int)

    return df


def predict_single(input_data: dict) -> dict:
    """Predict charges for a single input dict.

    Returns a dict with `predicted_price`, `confidence_interval`, and `prediction_time`.
    """
    model, preprocessor = _load_artifacts()

    df = pd.DataFrame([input_data])
    df = _add_derived_features(df)

    # Preprocess and predict
    X_processed = preprocessor.transform(df)
    pred = model.predict(X_processed)[0]
    pred = round(float(pred), 2)

    ci = [round(float(pred * 0.9), 2), round(float(pred * 1.1), 2)]

    return {
        "predicted_price": pred,
        "confidence_interval": ci,
        "prediction_time": datetime.now().isoformat(),
    }


def predict_batch(input_list: list) -> list:
    """Predict charges for a list of input dicts or a DataFrame-like list.

    Returns list of numeric predictions.
    """
    model, preprocessor = _load_artifacts()
    if isinstance(input_list, pd.DataFrame):
        df = input_list
    else:
        df = pd.DataFrame(input_list)

    df = _add_derived_features(df)
    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)
    return [round(float(p), 2) for p in preds.tolist()]


if __name__ == "__main__":
    # Quick local demo (will raise informative errors if artifacts missing)
    example = {
        "age": 29,
        "sex": "female",
        "bmi": 26.2,
        "children": 0,
        "smoker": "no",
        "region": "southeast",
    }
    print(predict_single(example))

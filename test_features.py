import pandas as pd
from insurance_pricing.features import FeatureEngineering


def main():
    # small sample to smoke-test the transformer
    df = pd.DataFrame([
        {'age': 25, 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'sex': 'female', 'charges': 2000},
        {'age': 45, 'bmi': 30.5, 'children': 2, 'smoker': 'yes', 'sex': 'male', 'charges': 12000},
        {'age': 60, 'bmi': 28.0, 'children': 1, 'smoker': 'no', 'sex': 'male', 'charges': 15000},
    ])

    fe = FeatureEngineering(verbose=True)
    df_out = fe.transform(df)
    print('\nTransformed shape:', df_out.shape)
    print(df_out.head().to_string())


if __name__ == '__main__':
    main()

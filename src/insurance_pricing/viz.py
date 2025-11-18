import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_bmi_vs_charges(df: pd.DataFrame) -> None:
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker")
    plt.title("BMI vs Charges by Smoker Status")
    plt.tight_layout()
    plt.show()

    
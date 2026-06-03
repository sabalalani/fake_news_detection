import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "data/processed/cleaned_news.csv"
)

sns.countplot(
    x="label",
    data=df
)

plt.savefig(
    "results/class_distribution.png"
)

df["length"] = df["content"].str.len()

sns.histplot(
    df["length"],
    bins=50
)

plt.savefig(
    "results/article_lengths.png"
)
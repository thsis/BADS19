import os
import seaborn as sns
import pandas as pd
from preprocessing.cleaning import clean
from preprocessing.features import FeatureGenerator

from matplotlib import pyplot as plt


if __name__ == "__main__":
    traindatapath = os.path.join("data", "BADS_WS1819_known.csv")
    known = clean(traindatapath)

    fg = FeatureGenerator()
    X_known = fg.fit_transform(known, "return")

    corr = X_known.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                cmap=plt.cm.coolwarm,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    fig.savefig("eda/corrplot.png")

    corr = pd.melt(corr.reset_index(), id_vars="index")
    idx = (corr.value.abs() > 0.85) & (corr["index"] != corr.variable)
    print(corr.loc[idx, :])
